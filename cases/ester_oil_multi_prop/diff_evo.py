"""
    Main script for diffusion-evolution.
    From MolSculptor: https://github.com/egg5154/MolSculptor
"""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import json
import numpy as np
import jax.numpy as jnp
import jax.tree_util as jtu
import pickle as pkl
import flax.linen as nn
import argparse
import datetime
import functools
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from pathlib import Path
from tqdm import tqdm
from logging import getLogger, Logger
from ml_collections import ConfigDict
from molsculptor.src.model.diffusion_transformer import DiffusionTransformer
from molsculptor.src.common.utils import safe_l2_normalize
from molsculptor.train.scheduler import GaussianDiffusion
from molsculptor.train.inference import InferEncoder, InferDecoder, Inferencer, smi2graph_features
from molsculptor.utils import nsga3_constrained_selection, expand_batch_dim, encoder_function, decoder_function, \
    has_substructure, find_repeats, sim_function, set_recoder, load_infer_config, \
    load_infer_params, set_infer_config
from molsculptor.utils import DiffusionTFG
from cases.ester_oil_multi_prop.qsar_predictors import MultiObjectiveQsarScorer

from ester_oil.selection import select as nsga3_constrained_selection

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATH_ARG_KEYS = {
    "dit_config_path",
    "dit_params_path",
    "vae_config_path",
    "vae_params_path",
    "alphabet_path",
}


def resolve_project_paths(args: dict) -> dict:
    for key in PATH_ARG_KEYS:
        if key in args:
            path = Path(args[key])
            if not path.is_absolute():
                args[key] = str((PROJECT_ROOT / path).resolve())
    return args

# def selection(scores: np.ndarray, npop: int, constraints: np.ndarray,):
#     constraints = (constraints < 0).prod(axis = 1)
#     scores = (scores.squeeze(-1) + 273.15) / 100.
#     w = scores * constraints
#     return np.argsort(w)[-npop:]
#     # return np.random.choice(np.arange(len(scores)), size = npop, replace = False, p = w / w.sum(),)

UNIQUE_SMILES = []
def infer(args: dict,):

    #################################################################################
    #               Setting constants, recoder and loading networks                 #
    #################################################################################

    #### set constants
    TOTAL_STEP = args['total_step']
    DEVICE_BATCH_SIZE = args['npops']
    N_TOKENS = args['num_latent_tokens']
    N_EQ_STEPS = args['eq_steps']
    N_REPLICATE = args['noffs']
    DIM = args['dim_latent']
    generation_mode = args.get('generation_mode', 'seeded')
    if generation_mode not in {'seeded', 'de_novo'}:
        raise ValueError(f'Unsupported generation_mode: {generation_mode}')
    os.makedirs(args['save_path'], exist_ok=True)

    rng_key = jax.random.PRNGKey(args['random_seed'])
    np.random.seed(args['random_seed'])

    recoder = set_recoder(args['logger_path'],)

    #### load net
    net_config, train_config, _, global_config = load_infer_config(args['dit_config_path'])
    vae_config, _, data_config, vae_global_config = load_infer_config(args['vae_config_path'])
    dit_net = DiffusionTransformer(net_config, global_config)
    scheduler = GaussianDiffusion(train_config,)
    encoding_net = InferEncoder(vae_config, vae_global_config)
    decoding_net = InferDecoder(vae_config, vae_global_config)

    #### load params
    params, encoder_params, decoder_params = load_infer_params(
        args['dit_params_path'], args['vae_params_path'],)
    
    #### set inferencer
    beam_size = args['beam_size']
    infer_config = set_infer_config(args, data_config,)
    inferencer = Inferencer(
        encoding_net, decoding_net, encoder_params, decoder_params, infer_config)
    
    diffusion = DiffusionTFG(
        config = ConfigDict(args['diffusion_config']),
        dit_net = dit_net, dit_params = params,
        scheduler = scheduler, loss_fn = None,
    )

    #################################################################################
    #                    Defining functions for searching steps                     #
    #################################################################################

    #### define encoder & decoder functions
    with open(args['alphabet_path'], 'rb') as f: # load alphabet
        alphabet: dict = pkl.load(f)
        alphabet = alphabet['symbol_to_idx']
    reverse_alphabet = {v: k for k, v in alphabet.items()}
    encoder_f = functools.partial(
        encoder_function, inferencer = inferencer)
    decoder_f = functools.partial(
        decoder_function, inferencer = inferencer, reverse_alphabet = reverse_alphabet, beam_size = beam_size)
    
    #### load QSAR objectives
    qsar_scorer = MultiObjectiveQsarScorer(
        args["qsar_objectives"],
        device=args.get("qsar_device"),
        batch_size=args.get("qsar_batch_size", 64),
    )
    head_name_list = [objective["name"] for objective in args["qsar_objectives"]]

    #### define reward functions & constraint functions
    def reward_function(smiles, cache_dir = args['cache_dir'],):
        scores, raw_predictions = qsar_scorer(smiles)
        pred_df = pd.DataFrame({"SMILES": smiles})
        for name in head_name_list:
            pred_df[name] = raw_predictions[name]
        return scores, pred_df

    def constraint_function(molecule_dict, scores = None,):
        global UNIQUE_SMILES
        constraints, constraint_scores = [], []

        if generation_mode == 'de_novo':
            decoded_valid = molecule_dict.get(
                'decoded_valid', np.ones(len(molecule_dict['smiles']), np.int32))
            constraint_scores.append(decoded_valid)
            constraints.append(decoded_valid * (-1) + 0.5)

        is_repeat, UNIQUE_SMILES = find_repeats(molecule_dict['smiles'], UNIQUE_SMILES)
        constraint_scores.append(is_repeat)
        constraints.append(is_repeat * (-1) + 0.5)

        is_substruct = has_substructure(molecule_dict['smiles'], args['substructure_smarts'])
        constraint_scores.append(is_substruct)
        constraints.append(is_substruct * (-1) + 0.5)

        if generation_mode == 'seeded':
            tanimoto_sim = sim_function(molecule_dict['smiles'], np.array([args['init_smiles'],] * len(molecule_dict['smiles']), object),)
            constraint_scores.append(tanimoto_sim)
            constraints.append((tanimoto_sim > args['tanimoto_threshold']) * (-1) + 0.5)

        return np.stack(constraints, axis=1), np.stack(constraint_scores, axis=1)

    def replicate_func(x):
        # (npop, ...) -> (npop, r, ...) -> (npop * r, ...)
        repeat_x = np.repeat(x[:, None], N_REPLICATE, axis = 1)
        return repeat_x.reshape(-1, *repeat_x.shape[2:])
    
    #################################################################################
    #                                 Search step                                   #
    #################################################################################

    def diffusion_evo_step(
            step_it: int, x: jax.Array, rng_key: jax.Array, config: ConfigDict, cached: dict):

        diffusion_time_it = config.time[step_it] # pyright: ignore[reportOptionalSubscript]
        """Decoding to molecules: {'graphs', 'smiles', 'latents',} 
        Shape of (npop * r, ...)
        """
        decode_molecules = decoder_f(
            x, replicate_func(cached['molecules'][-1]['smiles']), replicate_func(cached['molecules'][-1]['latents']),
            )
        
        ### scoring: (npop * r,)
        scores, pred_df = reward_function(decode_molecules['smiles'],) # (npop * r, m)
        constraints, constraint_scores = constraint_function(decode_molecules, scores) # (npop * r, c)
        # print('constraint check:', (constraints < 0).sum(0))
        
        """Evolution Algorithm step.
        We combined the parent population and offsprings.
        """
        scores = np.concatenate([cached['scores'][-1], scores], axis = 0) # (npop * r + npop, m)
        pred_df = pd.concat([cached['pred_df'][-1], pred_df], axis = 0).reset_index(drop=True)
        constraints = np.concatenate([cached['constraints'][-1], constraints], axis = 0) # (npop * r + npop, c)
        constraint_scores = np.concatenate([cached['constraint_scores'][-1], constraint_scores], axis = 0) # (npop * r + npop, c)
        decode_molecules = jtu.tree_map(
            lambda x, y: np.concatenate([x, y], axis = 0), cached['molecules'][-1], decode_molecules)
        # breakpoint()
        choiced_idx = nsga3_constrained_selection(scores, DEVICE_BATCH_SIZE, constraints,)
        choiced_molecules = jtu.tree_map(lambda x: x[choiced_idx], decode_molecules) ## (npop, ...)
        choiced_scores = scores[choiced_idx] ## (npop, m)
        choiced_pred_df = pred_df.iloc[choiced_idx].reset_index(drop=True)
        choiced_constraints = constraints[choiced_idx] ## (npop, c)
        choiced_constraint_scores = constraint_scores[choiced_idx] ## (npop, c)
        for _i in range(choiced_scores.shape[-1]):
            recoder.info(f'Top scores {_i}: {np.sort(choiced_scores[:, _i])[:4]}')
        
        """Updating cached info.
        """
        cached['molecules'].append(choiced_molecules)
        cached['scores'].append(choiced_scores)
        cached['constraint_scores'].append(choiced_constraint_scores)
        cached['constraints'].append(choiced_constraints)
        cached['pred_df'].append(choiced_pred_df)

        if (step_it + 1) == config.search_steps:
            return x, cached, rng_key

        """Get latents for next generation.
        Renoise and denoise with TFG.
        """
        ### encoding: (npop, npt, d) -> (npop * r, npt, d)
        choiced_x = encoder_f(choiced_molecules['graphs'])
        choiced_x = replicate_func(choiced_x)
        choiced_x *= jnp.sqrt(choiced_x.shape[-1]) # scale here

        ### renoise & denoise
        x_out, rng_key = diffusion.jit_noise(choiced_x, diffusion_time_it, rng_key)
        x_out, rng_key = diffusion.vanilla_denoise(
            x_out, diffusion_time_it, cached['mask'], cached['rope_index'], rng_key, cached['prompt_labels'], cached['force_drop_ids'])
        return x_out, cached, rng_key
    
    #################################################################################
    #                            Defining main function                             #
    #################################################################################

    def diffusion_evo_optimization(config, rng_key, init_smiles,):
        ### init_molecules: dict {'graphs', 'smiles',}, has batch dim.

        """Create initial scores, similarity & constraints.
        Note that we add repeat constraints here to avoid duplicates.
        """
        init_key, rng_key = jax.random.split(rng_key)
        init_molecules: dict = {
            'smiles': np.asarray(init_smiles, object),
            'graphs': smi2graph_features(init_smiles),
            'decoded_valid': np.asarray(1, np.int32),
        }
        init_molecules = expand_batch_dim(DEVICE_BATCH_SIZE, init_molecules)
        init_scores, init_pred_df = reward_function(init_molecules['smiles'],)
        x: jax.Array = encoder_f(init_molecules['graphs']) ## (npop, npt, dim)
        init_molecules['latents'] = np.array(x, np.float32)
        init_constraints, init_constraint_scores = constraint_function(jtu.tree_map(replicate_func, init_molecules), init_scores)
        init_constraints, init_constraint_scores = init_constraints[:DEVICE_BATCH_SIZE], init_constraint_scores[:DEVICE_BATCH_SIZE]
        
        x = x * jnp.sqrt(x.shape[-1]) # scale here
        x = replicate_func(x) ## (npop * r, npt, dim)
        init_t = (args['t_min'] + args['t_max']) // 2
        mask = jnp.ones((DEVICE_BATCH_SIZE * N_REPLICATE, N_TOKENS), jnp.int32)
        rope_index = \
            jnp.array(
                [np.arange(N_TOKENS),] * (DEVICE_BATCH_SIZE * N_REPLICATE), 
                dtype = jnp.int32).reshape(DEVICE_BATCH_SIZE * N_REPLICATE, N_TOKENS)
        
        """Generate first offsprings"""
        recoder.info(f'Generating init offsprings...')
        prompt_labels, force_drop_ids = diffusion.generate_input_labels(DEVICE_BATCH_SIZE, N_REPLICATE, args['prop_ranges'])
        x, rng_key = diffusion.jit_noise(x, init_t, init_key)
        x, rng_key = diffusion.vanilla_denoise(
            x, init_t, mask, rope_index, rng_key, prompt_labels, force_drop_ids)
        cached = {
            'mask': mask, 
            'rope_index': rope_index,
            'prompt_labels': prompt_labels,
            'force_drop_ids': force_drop_ids,
            'molecules': [init_molecules,],
            'scores': [init_scores],
            'pred_df': [init_pred_df,],
            'constraints': [init_constraints],
            'constraint_scores': [init_constraint_scores],
        }

        """Evolution loop"""
        recoder.info(f'Starting search, total steps = {config.search_steps}')
        for step in range(config.search_steps):
            recoder.info(f'----------------------------------------------------------------')
            recoder.info(f'Searching step {step + 1}, noise mutation steps {config.time[step]}')
            x, cached, rng_key = diffusion_evo_step(step, x, rng_key, config, cached)
        # recoder.info(f'----------------------------------------------------------------')

        return cached

    def de_novo_evo_optimization(config, rng_key):
        """Create generation 0 from pure noise, then run evolutionary search."""
        candidate_count = DEVICE_BATCH_SIZE * N_REPLICATE
        mask = jnp.ones((candidate_count, N_TOKENS), jnp.int32)
        rope_index = jnp.tile(jnp.arange(N_TOKENS, dtype=jnp.int32), (candidate_count, 1))
        prompt_labels, force_drop_ids = diffusion.generate_input_labels(
            DEVICE_BATCH_SIZE, N_REPLICATE, args['prop_ranges'])

        de_novo_t = int(args.get('de_novo_t', scheduler.num_timesteps - 1))
        if not 1 <= de_novo_t < scheduler.num_timesteps:
            raise ValueError(
                f'de_novo_t must be in [1, {scheduler.num_timesteps - 1}], got {de_novo_t}')

        recoder.info(
            f'Generating {candidate_count} de novo candidates from Gaussian noise '
            f'at diffusion step {de_novo_t}...')
        rng_key, noise_key = jax.random.split(rng_key)
        x = jax.random.normal(noise_key, (candidate_count, N_TOKENS, DIM), jnp.float32)
        for t in tqdm(range(de_novo_t, 0, -1), desc='De novo denoising'):
            x, rng_key = diffusion.jit_denoise_step(
                diffusion.dit_params, x, mask, t, rope_index, rng_key,
                prompt_labels, force_drop_ids)

        fallback_smiles = np.full(candidate_count, 'C', object)
        fallback_latents = np.zeros((candidate_count, N_TOKENS, DIM), np.float32)
        candidates = decoder_f(x, fallback_smiles, fallback_latents)
        scores, pred_df = reward_function(candidates['smiles'])
        constraints, constraint_scores = constraint_function(candidates, scores)
        chosen_idx = nsga3_constrained_selection(scores, DEVICE_BATCH_SIZE, constraints)

        init_molecules = jtu.tree_map(lambda value: value[chosen_idx], candidates)
        init_scores = scores[chosen_idx]
        init_pred_df = pred_df.iloc[chosen_idx].reset_index(drop=True)
        init_constraints = constraints[chosen_idx]
        init_constraint_scores = constraint_scores[chosen_idx]
        feasible_count = int(np.all(init_constraints < 0, axis=1).sum())
        recoder.info(
            f'De novo generation 0 selected: {DEVICE_BATCH_SIZE} molecules, '
            f'{feasible_count} satisfy all constraints.')

        selected_x = encoder_f(init_molecules['graphs']) * jnp.sqrt(DIM)
        selected_x = replicate_func(selected_x)
        init_t = (args['t_min'] + args['t_max']) // 2
        selected_x, rng_key = diffusion.jit_noise(selected_x, init_t, rng_key)
        selected_x, rng_key = diffusion.vanilla_denoise(
            selected_x, init_t, mask, rope_index, rng_key,
            prompt_labels, force_drop_ids)

        cached = {
            'mask': mask,
            'rope_index': rope_index,
            'prompt_labels': prompt_labels,
            'force_drop_ids': force_drop_ids,
            'molecules': [init_molecules],
            'scores': [init_scores],
            'pred_df': [init_pred_df],
            'constraints': [init_constraints],
            'constraint_scores': [init_constraint_scores],
        }

        recoder.info(f'Starting de novo search, total steps = {config.search_steps}')
        for step in range(config.search_steps):
            recoder.info('----------------------------------------------------------------')
            recoder.info(
                f'Searching step {step + 1}, noise mutation steps {config.time[step]}')
            selected_x, cached, rng_key = diffusion_evo_step(
                step, selected_x, rng_key, config, cached)
        return cached
    
    #################################################################################
    #                               Running search                                  #
    #################################################################################

    #### recoding info
    # args_dict = vars(args)
    recoder.info(f'=====================INPUT ARGS=====================')
    recoder.info("INPUT ARGS:")
    for k, v in args.items():
        recoder.info(f"\t{k}: {v}")
    
    time_schedule = np.random.randint(args['t_min'], args['t_max'], size = (TOTAL_STEP,))
    search_config = ConfigDict({
        'time': time_schedule, 'eq_steps': N_EQ_STEPS, 'search_steps': TOTAL_STEP,
        'constraint_weights': None,})
    infer_start_time = datetime.datetime.now()
    recoder.info(f'=====================START INFERENCE=====================')
    if generation_mode == 'de_novo':
        cached = de_novo_evo_optimization(search_config, rng_key)
    else:
        cached = diffusion_evo_optimization(search_config, rng_key, args['init_smiles'],)
    save_file = {
        'smiles': [mol['smiles'] for mol in cached['molecules']],
        'scores': cached['scores'],
        'pred_df': cached['pred_df'],
        'constraints': cached['constraints'],
        'constraint_scores': cached['constraint_scores'],
    }
    save_path = os.path.join(args['save_path'], f'diffusion_evo_opt.pkl')
    with open(save_path, 'wb') as f:
        pkl.dump(save_file, f)
    
    ## inference done
    recoder.info(f'=====================END INFERENCE=====================')
    tot_time = datetime.datetime.now() - infer_start_time
    recoder.info(f'Inference done, time {tot_time}, results saved to {args["save_path"]}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type = str, default = os.path.join(os.path.dirname(__file__), 'config.json'))
    parser.add_argument('--save_path', type = str, default = os.path.dirname(__file__),)
    ags = parser.parse_args()
    with open(ags.config_path, "r", encoding="utf-8") as f:
        args = json.load(f)
        args = resolve_project_paths(args)
        args['save_path'] = ags.save_path
        args['logger_path'] = os.path.join(ags.save_path, 'log.txt')
        args['cache_dir'] = os.path.join(ags.save_path, 'cache')
    infer(args)

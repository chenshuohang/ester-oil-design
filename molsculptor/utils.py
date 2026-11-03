import jax.tree_util as jtu
import numpy as np
import os
from rdkit import Chem
from .src.common.utils import safe_l2_normalize
from .train.inference import tokens2smiles, smi2graph_features, standardize, sanitize_smiles
from .train.rewards import LogP_reward, tanimoto_sim, QED_reward, SA_reward, admet_reward
from .train.dataloader import LABEL_MIN_MAX

def expand_batch_dim(batch_size: int, dict_array: dict):
    """
        dim expansion for diffes optimization.
    """
    def _fn(path, arr):
        if jtu.DictKey('atom_features') in path:
            ## check ndim == 2
            if arr.ndim == 2: # has batch dim
                return arr
            elif arr.ndim == 1: # no batch dim
                return arr[None].repeat(batch_size, axis = 0)
            else:
                raise
        elif jtu.DictKey('bond_features') in path:
            ## check ndim == 3
            if arr.ndim == 3:
                return arr
            elif arr.ndim == 2:
                return arr[None].repeat(batch_size, axis = 0)
            else:
                raise
        elif jtu.DictKey('scores') in path:
            ## check ndim == 1
            if arr.ndim == 2:
                return arr
            elif arr.ndim == 1:
                return arr[None].repeat(batch_size, axis = 0)
            else:
                raise
        else:
            ## check ndim == 1
            if arr.ndim == 1:
                return arr
            elif arr.ndim == 0:
                return arr[None].repeat(batch_size, axis = 0)
            else:
                raise
    
    return jtu.tree_map_with_path(_fn, dict_array)

### functions for reward calculation ###

def sim_function(smiles, init_smiles,):
        sim = np.asarray(tanimoto_sim(smiles, init_smiles), dtype = np.float32,)
        return sim
    
def has_substructure(smiles, sub_smarts = None):
    assert sub_smarts is not None, 'Input arg sub_smarts is None!'
    sub_m = Chem.MolFromSmarts(sub_smarts)
    search_ms = [Chem.MolFromSmiles(s) for s in smiles]
    has_substr = [m.HasSubstructMatch(sub_m) for m in search_ms]
    return np.array(has_substr, np.int32)

def find_repeats(smiles, unique_smiles: list):
    is_repeat = []
    for s in smiles:
        if s in unique_smiles:
            is_repeat.append(0)
        else:
            is_repeat.append(1)
            unique_smiles.append(s)
    return np.array(is_repeat, np.int32), unique_smiles

### functions for encoding and decoding ###
def encoder_function(graph_features, inferencer):
        return inferencer.jit_encoding_graphs(graph_features)

def decoder_function(latent_tokens, cached_smiles, cached_latent_tokens,
                     inferencer, reverse_alphabet, beam_size=4, n_minus_atms=0,):
        """
            For decoder g(S|z).
            NOTE: input latent tokens is scaled.
        """
        ## latent_tokens: (dbs, npt, d), cached_smiles: (dbs,)
        dbs, npt, d = latent_tokens.shape
        latent_tokens = safe_l2_normalize(
            latent_tokens / jnp.sqrt(d), axis = -1)
        output_latent_tokens = np.array(latent_tokens, jnp.float32)
        
        assert cached_smiles.shape[0] == dbs, f'{cached_smiles.shape[0]} != {dbs}'
        ## (dbs*bm, n_seq)
        output_tokens, aux = inferencer.beam_search(
            step = 0, cond = latent_tokens,)
        output_tokens = np.asarray(output_tokens, np.int32)
        ## (dbs*bm,) -> (dbs, bm)
        output_smiles = [
            tokens2smiles(t, reverse_alphabet) for t in output_tokens]
        output_smiles = np.asarray(
            output_smiles, object).reshape(dbs, beam_size)
        ## check if valid
        sanitized_output_smiles = np.empty((dbs,), object)
        decoded_valid = np.ones((dbs,), np.int32)
        count = 0
        for i_ in range(dbs):
            ## search for beam_size, from the most probable one
            ## if one is valid, then break.
            for j_ in range(beam_size - 1, -1, -1):
                smi_ = sanitize_smiles(output_smiles[i_, j_], n_minus_atms,) # standardize(output_smiles[i_, j_])
                if smi_:
                    sanitized_output_smiles[i_] = smi_
                    break
            ## if no one is valid, then use cached smiles.
            if sanitized_output_smiles[i_] is None:
                count += 1
                decoded_valid[i_] = 0
                sanitized_output_smiles[i_] = cached_smiles[i_]
                output_latent_tokens[i_] = cached_latent_tokens[i_]
                # output_latent_tokens = output_latent_tokens.at[i_].set(cached_latent_tokens[i_])
        print('validity:', (dbs - count) / dbs)
        print('uniqueness:', len(set(sanitized_output_smiles)) / dbs)
        ## make graph features
        sanitized_output_graph_features = [
            smi2graph_features(smi,) for smi in sanitized_output_smiles]
        batched_output_graph_features = {
            top_key: {sub_key:
                np.stack([this_graph[top_key][sub_key] for this_graph in sanitized_output_graph_features]) \
                    for sub_key in sanitized_output_graph_features[0][top_key].keys()
            } for top_key in ['atom_features', 'bond_features']
        }
        return {
            'graphs': batched_output_graph_features, 
            'smiles': sanitized_output_smiles,
            'latents': output_latent_tokens,
            'decoded_valid': decoded_valid,
            }

### functions for NSGA-II ###
def fast_non_dominated_sorting(scores, constraints, constraint_weights = None):
    ## scores: (N = n_pops, M = metrics)
    ## constraints: (N = n_pops, C = constraints), constraints are boolean values.

    # (N, C) -> (N,) -> (N, N)
    and_constraints_i = np.all(constraints, axis = 1).astype(np.int32)
    and_constraints_ij = and_constraints_i[:, None] + and_constraints_i[None, :]

    # cond 0: and_constraints_ij = 0
    # compare using constraints weighted sum
    # (N, C) @ (C,) -> (N,)
    n_, c_ = constraints.shape
    if constraint_weights is None: constraint_weights = np.ones(c_)
    weighted_constraints_i = constraints @ constraint_weights
    _compare_i_dom_j = (weighted_constraints_i[:, None] > weighted_constraints_i[None, :]).astype(np.int32)
    _compare_j_dom_i = (weighted_constraints_i[:, None] < weighted_constraints_i[None, :]).astype(np.int32)
    dominated_ij_0 = _compare_i_dom_j - _compare_j_dom_i

    # cond 1: and_constraints_ij = 1
    # compare using and constraints
    _compare_i_dom_j = (and_constraints_i[:, None] > and_constraints_i[None, :]).astype(np.int32)
    _compare_j_dom_i = (and_constraints_i[:, None] < and_constraints_i[None, :]).astype(np.int32)
    dominated_ij_1 = _compare_i_dom_j - _compare_j_dom_i

    # cond 2: and_constraints_ij = 2
    # compare using scores
    # (N, 1, M), (1, N, M) -> (N, N, M)
    _compare_i_dom_j = (scores[:, None] > scores[None, :]).astype(np.int32).prod(axis = 2)
    _compare_j_dom_i = (scores[:, None] < scores[None, :]).astype(np.int32).prod(axis = 2)
    dominated_ij_2 = _compare_i_dom_j - _compare_j_dom_i

    dominated_ij = dominated_ij_0 * (and_constraints_ij == 0).astype(np.int32) + \
        dominated_ij_1 * (and_constraints_ij == 1).astype(np.int32) + \
        dominated_ij_2 * (and_constraints_ij == 2).astype(np.int32)
    
    # (N, N) -> (N,)
    np_i = np.sum(dominated_ij == -1, axis = 1)
    # [(n_i,), ...]
    sp_i = [np.where(dominated_ij[_] == 1)[0] for _ in range(n_)]

    # get pareto fronts
    front_it = np.where(np_i == 0)[0]
    pareto_fronts = [front_it,]
    while front_it.size > 0:
        q_it = []
        for s in front_it:
            for q in sp_i[s]:
                np_i[q] -= 1
                if np_i[q] == 0: q_it.append(q)
        front_it = np.array(q_it, dtype = np.int32)
        pareto_fronts.append(front_it)

    return pareto_fronts

def crowding_distance_assignment(scores_it, inf = 1e6):
    ## sorting in one front set
    ## scores: (Ni, M)
    ## returns: (Ni,)

    n_i_, m_ = scores_it.shape
    distance = np.zeros(n_i_, dtype = np.float32)
    for _i in range(m_):
        front_scores_it = scores_it[:, _i] # (Ni,)
        sorted_index = np.argsort(front_scores_it)
        distance[sorted_index[0]] = inf
        distance[sorted_index[-1]] = inf
        # for i = 2 to Ni - 1
        sorted_scores = front_scores_it[sorted_index] # (Ni,)
        sorted_scores_ip1 = sorted_scores[2:]
        sorted_scores_im1 = sorted_scores[:-2]
        normed_distance_ = \
            (sorted_scores_ip1 - sorted_scores_im1) / (sorted_scores[-1] - sorted_scores[0] + 1e-12)
        distance[sorted_index[1: -1]] += normed_distance_ # (Ni - 2,)

    return distance

def NSGA_II(scores, constraints, constraint_weights = None, n_pops = 128, inf = 1e6):
    ## scores: (N = n_pops, M = metrics)
    ## constraints: (N = n_pops, C = constraints), constraints are boolean values.
    ## constraint_weights: (C,), weights for constraints.

    pareto_fronts = fast_non_dominated_sorting(scores, constraints, constraint_weights)
    next_pop_ids = np.empty(shape = (0,), dtype = np.int32)
    rank = 0
    ## strict sorting
    while next_pop_ids.size + pareto_fronts[rank].size <= n_pops:
        next_pop_ids = np.concatenate([next_pop_ids, pareto_fronts[rank]])
        rank += 1
    distance = crowding_distance_assignment(scores[pareto_fronts[rank]], inf)
    next_pop_ids = np.concatenate(
        [next_pop_ids, pareto_fronts[rank][np.argsort(distance)[::-1][:n_pops - next_pop_ids.size]]]
    )
    assert next_pop_ids.size == n_pops, \
        f'next_pop_ids.size = {next_pop_ids.size}, n_pops = {n_pops}'
    return next_pop_ids # (n_pops,)

import jax
import optax
import jax.numpy as jnp
import flax.linen as nn

from typing import Callable
from tqdm import tqdm
from ml_collections import ConfigDict
from .train.scheduler import _extract_into_tensor, GaussianDiffusion

class DiffusionTFG:
    """Training free guidance utils.
    """

    def __init__(self, config: ConfigDict, dit_net: nn.Module, dit_params: dict, 
                 scheduler: GaussianDiffusion, loss_fn: Callable = None,):

        self.config = config
        self.dit_params = dit_params
        self.dit_net = dit_net
        self.scheduler = scheduler

        """loss_fn is a wrapped function takes only x as input.
        example: loss_fn = partial(loss_net.apply, params = your_params, aux_input = your_aux_input)
        """
        if loss_fn is not None:
            self.loss_fn = loss_fn
            self.optimizer = optax.adam(
                learning_rate=config.learning_rate,
            )
            self.jit_grad_loss_fn = jax.jit(jax.grad(self.loss_fn, has_aux=True))
            self.jit_grad_loss_fn_xt = jax.jit(jax.grad(self.loss_fn_xt, has_aux=True))
            self.jit_optimizer_init = jax.jit(self.optimizer.init)

        ## tool functions
        self.jit_denoise_step = jax.jit(self.denoise_step)
        self.jit_noise_step = jax.jit(self.noise_step)
        self.jit_denoise = jax.jit(self.denoise)
        self.jit_noise = jax.jit(self.noise)
        self._jit_scale_fn = jax.jit(self._scale_fn)
        self._jit_x_update_fn = jax.jit(self._x_update_fn)
        self._jit_noise_with_eps = jax.jit(self._noise_with_eps)
        self._jit_pred_noise_from_xstart = jax.jit(self._pred_noise_from_xstart)
        self._jit_denoise_from_eps = jax.jit(self._denoise_from_eps)
        self._jit_denoise_step_from_eps = jax.jit(self._denoise_step_from_eps)

    def denoise_step(self, params, x, mask, time, rope_index, rng_key, labels = None, force_drop_ids = None):
        time = jnp.full((x.shape[0],), time) # (dbs,)
        eps_pred = self.dit_net.apply(
            {'params': params['params']['net']}, x, mask, time, 
            labels, force_drop_ids, rope_index) # (dbs, npt, d)
        mean, variance, log_variance = self.scheduler.p_mean_variance(
            x, time, eps_pred, clamp_x0_fn = None, clip = False)
        rng_key, sub_key = jax.random.split(rng_key)
        x = mean + jnp.exp(0.5 * log_variance) * jax.random.normal(sub_key, x.shape)
        return x, rng_key
    
    def noise_step(self, x, time, rng_key):
        time = jnp.full((x.shape[0],), time) # (dbs,)
        rng_key, sub_key = jax.random.split(rng_key)
        x = self.scheduler.q_sample_step(x, time, jax.random.normal(sub_key, x.shape))
        return x, rng_key
    
    def noise(self, x, time, rng_key):
        """q(x_t | x_0)"""
        time = jnp.full((x.shape[0],), time) # (dbs,)
        rng_key, sub_key = jax.random.split(rng_key)
        x = self.scheduler.q_sample(x, time, jax.random.normal(sub_key, x.shape))
        return x, rng_key

    def denoise(self, params, x, mask, time, rope_index, labels = None, force_drop_ids = None):
        """p(x_0 | x_t)"""
        time = jnp.full((x.shape[0],), time)
        eps_pred = self.dit_net.apply(
            {'params': params['params']['net']}, x, mask, time, 
            labels, force_drop_ids, rope_index) # (dbs, npt, d)
        x_0 = self.scheduler._predict_xstart_from_eps(x, time, eps_pred)
        return x_0, eps_pred
    
    def _denoise_from_eps(self, eps, x, time):
        time = jnp.full((x.shape[0],), time)
        x_0 = self.scheduler._predict_xstart_from_eps(x, time, eps)
        return x_0
    
    def _x_update_fn(self, opt_state, x,):
        """Update function for optimizer.
        NOTE: input x is scaled."""

        grad, _ = jax.grad(self.loss_fn, has_aux=True)(x,) # TODO: check here
        updates, opt_state = self.optimizer.update(grad, opt_state, x)
        x = optax.apply_updates(x, updates)
        return x, grad, opt_state
    
    def _scale_fn(self, time_step,):
        """Scale function for forward guidance."""
        # search_step_scale = 1 / (1 + jnp.exp(-0.5 * (search_step - 15))) # TODO: test here
        # search_step_scale = 1 # debug
        return self.scheduler.sqrt_one_minus_alphas_cumprod[time_step] * 1e-2
    
    def _optx(self, x, opt_state):
        """NOTE: x is scaled(norm=dim).
        params is for the loss function, and x will be optimized."""
        x, grad_save, opt_state = self._jit_x_update_fn(opt_state, x,) # save init grad
        for _ in range(self.config.optimization_steps - 1):
            x, _, opt_state = self._jit_x_update_fn(opt_state, x,)
        return x, grad_save
    
    def _denoise_step_from_eps(self, x, eps, time, rng_key):
        time = jnp.full((x.shape[0],), time) # (dbs,)
        mean, var, log_var = self.scheduler.p_mean_variance(
            x, time, eps, clamp_x0_fn=None, clip=False)
        rng_key, sub_key = jax.random.split(rng_key)
        x = mean + jnp.exp(0.5 * log_var) * jax.random.normal(sub_key, x.shape)
        return x, rng_key
    
    def _pred_noise_from_xstart(self, x, time, x_0):
        time = jnp.full((x.shape[0],), time) # (dbs,)
        return (
            _extract_into_tensor(self.scheduler.sqrt_recip_alphas_cumprod, time, x.shape) * x
            - x_0
        ) / _extract_into_tensor(self.scheduler.sqrt_recipm1_alphas_cumprod, time, x.shape)
    
    def _noise_with_eps(self, x_0, time, eps):
        time = jnp.full((x_0.shape[0],), time) # (dbs,)
        """q(x_t | x_0)"""
        coeff1 = _extract_into_tensor(self.scheduler.sqrt_alphas_cumprod, time, x_0.shape)
        coeff2 = _extract_into_tensor(self.scheduler.sqrt_one_minus_alphas_cumprod, time, x_0.shape)

        return x_0 * coeff1 + eps * coeff2
    
    def loss_fn_xt(self, x_t, mask, time, rope_index, labels, force_drop_ids):
        x_0, eps = self.jit_denoise(self.dit_params, x_t, mask, time, rope_index, labels, force_drop_ids)
        out, _ = self.loss_fn(x_0,)
        return out, eps
    
    def tfg_step(self, x_t, time, mask, rope_index, rng_key, 
                 labels=None, force_drop_ids=None, renoise=True):

        g_0, eps = self.jit_grad_loss_fn_xt(x_t, mask, time, rope_index, labels, force_drop_ids)
        eps += self._jit_scale_fn(time) * g_0
        x_0_forward = self._jit_denoise_from_eps(eps, x_t, time)
        opt_state = self.jit_optimizer_init(x_0_forward)
        x_0_backward, _ = self._optx(x_0_forward, opt_state)
        # ## debug scores
        # _, (ss, ls) = self.loss_fn(x_0_backward,)
        # print(jtu.tree_map(lambda arr: (arr.max(), arr.min()), ls))
        eps_hat = self._jit_pred_noise_from_xstart(x_t, time, x_0_backward)
        x_tminus1 = self._jit_noise_with_eps(x_0_backward, time - 1, eps_hat)
        """time range: (T, T-1, ...., 1)
        """
        if renoise:
            x_t, rng_key = self.jit_noise_step(x_tminus1, time, rng_key)
            return x_t, rng_key
        else:
            return x_tminus1, rng_key
    
    def tfg_denoise(
            self, x_out, diffusion_time_it, mask, 
            rope_index, rng_key, labels = None, force_drop_ids = None):

        for t_i in tqdm(range(diffusion_time_it)):
            t = diffusion_time_it - t_i
            # we run some eq steps first for efficient sampling
            for _ in range(self.config.eq_steps):
                x_out, rng_key = self.tfg_step(
                    x_out, t, mask, rope_index, rng_key, labels, force_drop_ids)
            # x: (dbs*r, npt, d)
            x_out, rng_key = self.tfg_step(
                x_out, t, mask, rope_index, rng_key, labels, force_drop_ids, renoise=False)
        return x_out, rng_key
    
    def vanilla_denoise(
            self, x, diffusion_time_it, mask, rope_index, 
            rng_key, labels = None, force_drop_ids = None):
        
        for t_i in tqdm(range(diffusion_time_it)):
            t = diffusion_time_it - t_i
            ### we run some eq steps first for efficient sampling
            for eq_step in range(self.config.eq_steps):
                x, rng_key = self.jit_denoise_step(self.dit_params, x, mask, t, rope_index, rng_key, labels, force_drop_ids)
                x, rng_key = self.jit_noise_step(x, t, rng_key)
            ### x: (n_device, dbs, npt, d)
            x, rng_key = self.jit_denoise_step(self.dit_params, x, mask, t, rope_index, rng_key, labels, force_drop_ids)
        return x, rng_key
    
    def generate_input_labels(self, npops: int, nreps: int, prop_ranges: dict):
        decimal_dict = {
            'n_double_bonds': 0, 'n_unsaturation': 0,
            'mw': 2, 'n_rings': 0, 'logp': 2, 'n_rot_bonds': 0,
        }
        input_labels = {}
        for label, (label_min, label_max) in LABEL_MIN_MAX.items():
            _range = prop_ranges.get(label)
            if _range is None:
                input_labels[label] = jnp.full(npops*nreps, -1e3)
            else:
                input_labels[label] = (jnp.tile(
                    jnp.round(jnp.linspace(*_range, npops), decimal_dict[label]), nreps) - label_min) / (label_max - label_min)
        
        force_drop_ids = jnp.array([0,] * (npops*nreps), jnp.int32) ## no drop
        return input_labels, force_drop_ids


import copy
import math
import datetime
import pickle as pkl
import logging
from logging import Logger

def set_recoder(logger_path: str) -> Logger:

    #### set recoder
    recoder = logging.getLogger("inferencing dit")
    recoder.setLevel(level = logging.DEBUG)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level = logging.DEBUG)
    recoder.addHandler(stream_handler)
    file_handler = logging.FileHandler(logger_path)
    file_handler.setLevel(level = logging.DEBUG)
    recoder.addHandler(file_handler)
    recoder.propagate = False

    return recoder

from typing import Tuple
def load_infer_config(config_path: str) -> Tuple[ConfigDict, ConfigDict, ConfigDict, ConfigDict]:
    with open(config_path, 'rb') as f:
        config_dicts = pkl.load(f)
    global_config = ConfigDict(config_dicts['global_config'])
    net_config = ConfigDict(config_dicts['net_config'])
    train_config = ConfigDict(config_dicts['train_config'])
    data_config = ConfigDict(config_dicts['data_config'])
    global_config.dropout_flag = False
    return net_config, train_config, data_config, global_config

def load_infer_params(dit_params_path: str, vae_params_path: str) -> Tuple[dict, dict, dict]:
    with open(vae_params_path, 'rb') as f:
        vae_params = pkl.load(f)
        vae_params = jtu.tree_map(lambda x: jnp.asarray(x), vae_params)
    encoder_params = {
        'Encoder_0': vae_params['params']['generator']['Encoder_0'],
        'Dense_0': vae_params['params']['generator']['Dense_0'],
    }
    decoder_params = {
        'Decoder_0': vae_params['params']['generator']['Decoder_0'],
    }
    with open(dit_params_path, 'rb') as f:
        params = pkl.load(f)
        params = jtu.tree_map(jnp.asarray, params)
    return params, encoder_params, decoder_params

from argparse import ArgumentParser
def set_infer_config(args: dict, data_config: ConfigDict,) -> ConfigDict:
    default_infer_config = {
        'sampling_method': args['sampling_method'],
        'device_batch_size': args['npops'],
        'n_seq_length': data_config['n_pad_token'],
        'beam_size': args['beam_size'],
        'bos_token': data_config['bos_token_id'],
        'eos_token': data_config['eos_token_id'],
        'n_local_device': jax.local_device_count(),
        'num_prefix_tokens': data_config['n_query_tokens'],
        'step_limit': 256,
    }
    return ConfigDict(default_infer_config)

def find_coupling_sites(mol: Chem.Mol):
    """
    Find potential coupling sites in a molecule.
    """
    matching_indices = []
    Chem.SanitizeMol(mol)

    for bond in mol.GetBonds():
        
        bond: Chem.Bond
        begin_atom: Chem.Atom = bond.GetBeginAtom()
        end_atom: Chem.Atom = bond.GetEndAtom()
        # 1. must be conjugated
        if not bond.GetIsConjugated():
            continue
            
        # 2. must be C or N
        valid_atomic_nums = {6, 7}  # C 和 N
        atomic_nums = [begin_atom.GetAtomicNum(), end_atom.GetAtomicNum()]
        if not any(num in valid_atomic_nums for num in atomic_nums):
            continue
        
        for atom in (begin_atom, end_atom):
            # 3. at least one hydrogen attached
            if atom.GetTotalNumHs() > 0:
                matching_indices.append(atom.GetIdx())
                
    return list(set(matching_indices))

def link_molecules(mol1: Chem.Mol, mol2: Chem.Mol, 
                   anchor_idx1: int, anchor_idx2: int, 
                   bond_type=Chem.rdchem.BondType.SINGLE):
    """
    Create a chemical bond between specified anchor points of two molecules.

    Assumptions:
    1. Both anchor atoms (anchor1, anchor2) have at least one H.
    2. The coupling reaction will remove one H from anchor1 and one H from anchor2.
    """

    combined_mol = Chem.CombineMols(mol1, mol2)
    editable_mol = Chem.RWMol(combined_mol)

    # compute new index for anchor_idx2 in the combined molecule
    num_atoms_mol1 = mol1.GetNumAtoms()
    new_anchor_idx2 = anchor_idx2 + num_atoms_mol1
    
    # update explicit H counts for anchor atoms
    atom1 = editable_mol.GetAtomWithIdx(anchor_idx1)
    atom2 = editable_mol.GetAtomWithIdx(new_anchor_idx2)
    atom1.SetNumExplicitHs(atom1.GetTotalNumHs() - 1)
    atom2.SetNumExplicitHs(atom2.GetTotalNumHs() - 1)

    # (optional, but recommended) tell RDKit not to recalculate the implicit Hs for these atoms
    atom1.SetNoImplicit(True)
    atom2.SetNoImplicit(True)

    # add the new bond and get the updated molecule
    editable_mol.AddBond(anchor_idx1, new_anchor_idx2, order=bond_type)
    final_mol = editable_mol.GetMol()
    
    # sanitize the final molecule (important)
    try:
        Chem.SanitizeMol(final_mol)
        return final_mol
    
    except Exception as e:
        return None

try:
    from pymoo.algorithms.moo.unsga3 import UNSGA3
    from pymoo.core.survival import Survival
    from pymoo.core.problem import Problem
    from pymoo.core.population import Population
    from pymoo.util.ref_dirs import get_reference_directions

    class ConstrProblem(Problem):
        def __init__(self, n_obj: int, n_ieq: int = 0, n_eq: int = 0):
            super().__init__(n_var=1, n_obj=n_obj, n_ieq_constr=n_ieq, n_eq_constr=n_eq, xl=0.0, xu=1.0)

        def _evaluate(self, X, out, *args, **kwargs):
            pass

    def nsga3_constrained_selection(F: np.ndarray, npop: int, G: np.ndarray = None, n_partitions: int = None):
        n, m = F.shape
        assert npop <= n
        if n_partitions is None:
            for p in range(1, 100):
                hp = math.comb(p + m - 1, m - 1)
                if hp > npop:
                    n_partitions = p - 1
                    break

        ref_dirs = get_reference_directions("das-dennis", m, n_partitions=n_partitions)
        algorithm = UNSGA3(ref_dirs=ref_dirs, pop_size=npop)
        X = np.arange(n).reshape(-1, 1)
        fields = {"X": X, "F": F}
        if G is not None:
            fields["G"] = G
            n_ieq = G.shape[1]
        else:
            n_ieq = 0
        pop = Population.new(**fields)
        problem = ConstrProblem(n_obj=m, n_ieq=n_ieq, n_eq=0)
        selector: Survival = algorithm.survival
        survivors = selector.do(problem=problem, pop=pop, n_survive=npop, algorithm=algorithm)
        return np.array([ind.X.item() for ind in survivors])

except ModuleNotFoundError:
    def nsga3_constrained_selection(F: np.ndarray, npop: int, G: np.ndarray = None, n_partitions: int = None):
        """
        Fallback selector when pymoo is unavailable.

        This reuses the repository's internal NSGA-II implementation and keeps
        the same "lower is better" semantics as the pymoo-based selector by
        negating the objective values before selection.
        """
        n, _ = F.shape
        assert npop <= n

        if G is None:
            feasible = np.ones((n, 1), dtype=bool)
        else:
            feasible = G < 0

        return NSGA_II(-F, feasible, n_pops=npop)

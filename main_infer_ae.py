"""
    main function for inferencing.
    saved info:
        1. graph latent features;
        2. infer output smiles;
        3. true canonical smiles;
        4. zinc ids;
"""

import os
import sys
import jax
import math
import logging
import datetime
import jax.numpy as jnp
import pickle as pkl
import numpy as np
import jax.tree_util as jtu
import argparse

def arg_parse():

    parser = argparse.ArgumentParser(description='Inputs for main.py')
    parser.add_argument('--name_list_path', type=str, help="path for loading name list")
    parser.add_argument('--logger_path', type=str, help="path for save logs")
    parser.add_argument('--params_path', type=str, help="path for loading params")
    parser.add_argument('--config_path', type=str, help='Location of config file.')
    parser.add_argument('--alphabet_path', type=str, help='Location of alphabet file.')
    parser.add_argument('--beam_size', type=int, default=1, help="search beam size")
    parser.add_argument('--sampling_method', type=str, default='beam', help="sampling method for inferencing")
    parser.add_argument('--device_batch_size', type=int, default=8, help="number of samples per device")
    parser.add_argument('--save_steps', type=int, default=10000, help="save ckpt steps")
    parser.add_argument('--start_infer_index', type=int, default=0, help="start infer index")
    parser.add_argument('--pre_load_steps', type=int, default=10, help="pre load step interval")
    parser.add_argument('--save_ckpt_path', type=str, default="./protoken_params/train_vq", help='Location of checkpoint file.')
    parser.add_argument('--sampler_config_path', type=str, default=None, help='Location of sampler config file.')

    ## distributed
    parser.add_argument('--coordinator_address', type=str, help="coordinator address")
    parser.add_argument('--num_processes', type=int, default=1, help="number of processes")
    parser.add_argument('--device_ids', nargs='+', type=int, default=None, help='local device ids')
    parser.add_argument('--rank', type=int, default=0, help="rank")

    arguments = parser.parse_args()
    
    return arguments

args = arg_parse()

# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".05" ## debug
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"
if args.num_processes > 1:
    jax.distributed.initialize(
        coordinator_address = args.coordinator_address, #"128.5.20.2:8888",
        num_processes = args.num_processes,
        process_id = args.rank,
        local_device_ids = args.device_ids,
    )

RANK = jax.process_index()
print("This rank:", RANK)

### import modules
from rdkit import Chem
from rdkit.rdBase import BlockLogs
from tqdm import tqdm
from Levenshtein import distance
from jax.tree_util import tree_map
from jax.experimental.multihost_utils import process_allgather
from ml_collections.config_dict import ConfigDict
from molsculptor.train.inference import InferEncoder, InferDecoder, Inferencer
from molsculptor.train.dataloader import read_files_in_parallel, make_feature

def load_ckpt(path):
    with open(path, 'rb') as f:
        params = pkl.load(f)
        params = tree_map(jnp.asarray, params)
    return params

def save_ckpt(path, params):
    ## save as numpy array
    with open(path, 'wb') as f:
        pkl.dump(tree_map(np.array, params), f)

def tokens2smiles(tokens, reverse_alphabet):

    ### tokens: (n_seq,)
    smiles = []
    for t in tokens[1:]: ## skip BOS
        if reverse_alphabet[t] == '[EOS]':
            break
        smiles.append(reverse_alphabet[t])
    return ''.join(smiles)

class InferDataLoader:

    def __init__(self, name_list, data_config,):

        self.name_list = name_list
        self.data_config = data_config
        self.process_id = jax.process_index()
        self.load_int = data_config['pre_load_int']

        ### create indexes and shuffle
        self.num_data = self.name_list.shape[0]
        np.random.seed(data_config['seed'])

        self.device_batch_size = data_config['batch_size_device']
        self.num_local_devices = jax.local_device_count()
        self.gbs = self.device_batch_size * jax.device_count()
        self.lbs = self.device_batch_size * jax.local_device_count()

    def organize(self, step_it):

        ### get indexes: (load_int * dbs * n_device)
        def _get_idxs():
            start_it = step_it * self.gbs + self.lbs * self.load_int * self.process_id
            stop_it = step_it * self.gbs + self.lbs * self.load_int * (self.process_id + 1)
            return np.arange(self.num_data)[np.arange(start_it, stop_it) % self.num_data]

        _idx = _get_idxs()
        # breakpoint() ## check here
        return self.name_list[_idx], _idx
    
    def _make_feature(self, name_list_trunk, data_config):

        #### make batched data
        data_from_pickle = read_files_in_parallel(name_list_trunk, data_config['n_workers'])
        graph_feat, sequence_feat = make_feature(data_from_pickle, data_config,)
        input_feat = {
            'graph_features': graph_feat,
            'tokens': np.array(
                sequence_feat['input_tokens'], 
                data_config['dtype']).squeeze(),
        }
        # breakpoint() ## check here
        return input_feat

    def load_data(self, step_it,):

        name_list_trunk, _idx = self.organize(step_it)
        data_config = self.data_config
        input_feat = self._make_feature(name_list_trunk, data_config)
        input_feat['ids'] = _idx
        return input_feat
    
    def select_data(self, data, data_it):

        pid = self.process_id
        start_idx = data_it * self.lbs
        stop_idx = (data_it + 1) * self.lbs
        data_it = tree_map(lambda arr: arr[start_idx: stop_idx], data)
        return data_it

#############################################
### function for batched data inferencing ###
#############################################
def inference_batched_data():

    ## set recoder
    recoder = logging.getLogger("inferencing")
    recoder.setLevel(level=logging.DEBUG)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level=logging.DEBUG) 
    recoder.addHandler(stream_handler)
    file_handler = logging.FileHandler(args.logger_path)
    file_handler.setLevel(level=logging.DEBUG)
    recoder.addHandler(file_handler)

    ## load name list
    with open(args.name_list_path, 'rb') as f:
        name_list: np.ndarray = pkl.load(f)['path']
    assert name_list.ndim == 1, \
        f"Error: name list should be 1-dim, but got {name_list.shape}."

    ## set basic constants
    DEVICE_BATCH_SIZE = args.device_batch_size
    N_LOCAL_DEVICES = jax.local_device_count()
    N_GLOBAL_DEVICES = jax.device_count()
    GLOBAL_BATCH_SIZE = N_GLOBAL_DEVICES * DEVICE_BATCH_SIZE
    LOCAL_BATCH_SIZE = N_LOCAL_DEVICES * DEVICE_BATCH_SIZE

    N_DATAS = name_list.shape[0]
    N_STEPS = N_DATAS // GLOBAL_BATCH_SIZE
    pre_load_steps = args.pre_load_steps

    ## basic info
    if (RANK == 0):
        recoder.info("DATA INFO:")
        recoder.info("\tGLOBAL BATCH_SIZE: {}".format(GLOBAL_BATCH_SIZE))
        recoder.info("\tNDATAS: {}".format(N_DATAS))
    
    ## load config & params
    with open(args.params_path, 'rb') as f:
        params = pkl.load(f)
        params = jtu.tree_map(lambda x: jnp.asarray(x), params)
    with open(args.config_path, 'rb') as f:
        config_dicts = pkl.load(f)
    global_config = ConfigDict(config_dicts['global_config'])
    net_config = ConfigDict(config_dicts['net_config'])
    data_config = config_dicts['data_config']
    data_config['n_repeat_k'] = 1 ### only load canonical SMILES
    data_config['pre_load_int'] = pre_load_steps
    data_config['batch_size_device'] = DEVICE_BATCH_SIZE
    global_config['dropout_flag'] = False
    
    encoding_net = InferEncoder(net_config, global_config)
    decoding_net = InferDecoder(net_config, global_config)

    ### init sampler
    beam_size = args.beam_size
    default_sampler_config = {
        'sampling_method': args.sampling_method,
        'device_batch_size': args.device_batch_size,
        'n_seq_length': data_config['n_pad_token'],
        'beam_size': beam_size,
        'bos_token': data_config['bos_token_id'],
        'eos_token': data_config['eos_token_id'],
        'n_local_device': jax.local_device_count(),
        'num_prefix_tokens': data_config['n_query_tokens'],
        'step_limit': 256,
    }
    if args.sampler_config_path:
        with open(args.sampler_config_path, 'rb') as f:
            sampler_config = pkl.load(f)
    else:
        sampler_config = default_sampler_config
    sampler_config = ConfigDict(sampler_config)
    encoder_params = {
        'Encoder_0': params['params']['generator']['Encoder_0'],
        'Dense_0': params['params']['generator']['Dense_0'],
    }
    decoder_params = {
        'Decoder_0': params['params']['generator']['Decoder_0'],
    }
    my_sampler = Inferencer(
        encoding_net, decoding_net, encoder_params, decoder_params, sampler_config)
    encoding_fn = my_sampler.pmap_encoding_graphs
    sample_fn = my_sampler.beam_search_multi_device ## we use beam search here

    ### set constants
    assert args.save_steps % pre_load_steps == 0, \
        "Error: save_steps % pre_load_steps != 0"
    start_infer_index = args.start_infer_index

    N_INFER_DATAS = N_DATAS - start_infer_index
    N_INFER_STEPS = math.ceil(N_INFER_DATAS / GLOBAL_BATCH_SIZE)
    GBS = GLOBAL_BATCH_SIZE
    LBS = LOCAL_BATCH_SIZE

    ### set alphabet
    with open(args.alphabet_path, 'rb') as f:
        alphabet: dict = pkl.load(f)
        alphabet = alphabet['symbol_to_idx']
    reverse_alphabet = {v: k for k, v in alphabet.items()}

    ### set dataloader
    dataloader = InferDataLoader(name_list[start_infer_index:], data_config)
    start_infer_step = start_infer_index // GLOBAL_BATCH_SIZE

    ## inferencing
    if (RANK == 0):
        recoder.info("\tN_INFER_DATAS: {}".format(N_INFER_DATAS))
        recoder.info("\tN_INFER_STEPS: {}".format(N_INFER_STEPS))
        recoder.info("\tSTART_INFER_INDEX: {}".format(start_infer_index))
        recoder.info("=====================START INFERENCING=====================")
    npt = data_config['n_query_tokens']
    d = net_config['latent_dim']
    latent_cached = np.empty((0, npt, d), np.float32)
    pred_smiles_cached = np.empty((0, beam_size), object)
    true_smiles_cached = np.empty((0,), object)
    id_cached = np.empty((0,), object)
    for _step in tqdm(range(N_INFER_STEPS)):

        if (_step % pre_load_steps == 0):
            start_time_load = datetime.datetime.now()
            pre_load_data_dict = dataloader.load_data(_step)
            end_time_load = datetime.datetime.now()
            data_it = 0
            pre_load_time = end_time_load - start_time_load

        data_dict = dataloader.select_data(pre_load_data_dict, data_it)
        data_it += 1

        input_graph_features = tree_map(lambda x: x.reshape(N_LOCAL_DEVICES, DEVICE_BATCH_SIZE, *x.shape[1:]), data_dict['graph_features'])
        latents = encoding_fn(input_graph_features)
        pred_tokens, cached_arr = sample_fn(step = 0, cond = latents) ## (ndev, dbs*bm, n_seq)
        pred_tokens = np.reshape(
            np.asarray(pred_tokens, np.int16), (LBS * beam_size, -1))
        pred_tokens = process_allgather(pred_tokens, True) ## (GBS, n_seq)
        pred_smiles = [
            tokens2smiles(t, reverse_alphabet) for t in pred_tokens]
        pred_smiles = np.asarray(pred_smiles, object).reshape(GBS, beam_size)

        true_tokens = data_dict['tokens'].reshape(LBS, -1) ## (ndev*dbs, n_seq)
        true_tokens = process_allgather(true_tokens, True) ## (GBS, n_seq)
        # breakpoint() ## check here
        true_smiles = [
            tokens2smiles(t, reverse_alphabet) for t in true_tokens]
        true_smiles = np.asarray(true_smiles, object).reshape(GBS,)

        ## (ndev, dbs*bm, npt, d) -> (ndev*dbs, npt, d)
        latent = np.asarray(cached_arr['cond']).reshape(LBS, beam_size, npt, d)[:, 0]
        latent = process_allgather(latent, True) ## (GBS, npt, d)
        ## (ndev, dbs) -> (ndev*dbs,)
        ids = np.asarray(data_dict['ids']).reshape(-1)
        ids = process_allgather(ids, True) ## (GBS,)

        ## concat
        pred_smiles_cached = np.concatenate((pred_smiles_cached, pred_smiles), axis = 0,)
        true_smiles_cached = np.concatenate((true_smiles_cached, true_smiles), axis = 0,)
        latent_cached = np.concatenate((latent_cached, latent), axis = 0,)
        id_cached = np.concatenate((id_cached, ids), axis = 0,)

        ## save
        if ((_step + 1) % args.save_steps == 0):
            if (RANK == 0):
                # print(f"Save for step: {_step + 1}")
                save_path = os.path.join(
                    args.save_ckpt_path, f"infer_{start_infer_step + _step + 1}.pkl")
                save_f = {
                    'pred_smiles': pred_smiles_cached,
                    'true_smiles': true_smiles_cached,  
                    'latent': latent_cached,
                    'ids': id_cached,
                }
                save_ckpt(save_path, save_f)

            ## reset cached dir
            pred_smiles_cached = np.empty((0, beam_size), object)
            true_smiles_cached = np.empty((0,), object)
            latent_cached = np.empty((0, npt, d), np.float32)
            id_cached = np.empty((0,), object)
    ## save for tail steps
    if (pred_smiles_cached.size > 0) and (RANK == 0):
        save_path = os.path.join(
            args.save_ckpt_path, f"infer_{start_infer_step + _step + 1}.pkl")
        save_f = {
            'pred_smiles': pred_smiles_cached,
            'true_smiles': true_smiles_cached,  
            'latent': latent_cached,
            'ids': id_cached,
        }
        save_ckpt(save_path, save_f)

if __name__ == "__main__":
    
    inference_batched_data()
    print("Inference Done.")
    jax.distributed.shutdown() if args.num_processes > 1 else None


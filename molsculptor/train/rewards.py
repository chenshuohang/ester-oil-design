"""
    In this script, we store the reward functions for guided diffusion.
    NOTE: Higher score is better!!!
"""

import jax
import numpy as np
import pandas as pd
import jax.numpy as jnp
import os
import time
import sys
import subprocess
import datetime
import multiprocessing

from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Crippen, QED, AllChem, DataStructs
from rdkit.Contrib.SA_Score import sascorer # type: ignore
# from .sascorer import calculateScore

def LogP_reward(molecule_dict):
    smiles = molecule_dict['smiles']
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    return [Crippen.MolLogP(m) for m in mols]

def QED_reward(smiles):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    return [QED.qed(m) for m in mols]

def SA_reward(smiles):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    return [sascorer.calculateScore(m) for m in mols]

def tanimoto_sim(smiles1, smiles2):
    scores = []
    assert smiles1.shape[0] == smiles2.shape[0]
    for i in range(smiles1.shape[0]):
        mol1 = Chem.MolFromSmiles(smiles1[i], sanitize = True)
        mol2 = Chem.MolFromSmiles(smiles2[i], sanitize = True)
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits = 2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits = 2048)
        score = DataStructs.FingerprintSimilarity(fp1, fp2)
        scores.append(score)
    return scores

# check mols because the rdkit in admetai is a previous version
# and some molecules cannot be parsed correctly
def _admet_rdkit_check(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        assert mol is not None
        mol.UpdatePropertyCache(strict=False)
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() in [15,]: # now only P
                charge = atom.GetFormalCharge()
                explicit_val = atom.GetExplicitValence()
                
                # the maximum allowed valence for P is 4
                if charge == 1 and explicit_val > 4:
                    assert False

        return True
    except Exception as e:
        return False 

def admet_reward(smiles, save_root, cmd_path,) -> pd.DataFrame:
    data_path = os.path.join(save_root, 'smiles.csv')
    mask_smiles = np.array([_admet_rdkit_check(s) for s in smiles])
    smiles = np.where(mask_smiles, smiles, 'C')
    smiles_df = pd.DataFrame({'smiles': smiles})
    smiles_df.to_csv(data_path, index=False)
    save_path = os.path.join(save_root, 'admet_pred.csv')
    subprocess.run(
        [cmd_path, '--data_path', data_path, '--save_path', save_path, '--smiles_column', 'smiles'],
        stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL, check = True,
    )
    pred_df = pd.read_csv(save_path)
    pred_df['mask'] = mask_smiles # true means valid molecule
    return pred_df
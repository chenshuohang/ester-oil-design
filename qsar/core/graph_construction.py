from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .rdkit_descriptors import (
        ID_COL,
        SMILES_COL,
        TARGET_COL,
        extract_numeric,
        pick_first_available,
        read_table,
    )
except ImportError:
    from rdkit_descriptors import (
        ID_COL,
        SMILES_COL,
        TARGET_COL,
        extract_numeric,
        pick_first_available,
        read_table,
    )


SMILES_CANDIDATES = [SMILES_COL, "SMILES(2D)", "smiles", "Smiles", "canonical_smiles"]
TARGET_CANDIDATES = [TARGET_COL, "Flash point", "flash_point", "闭口闪点℃", "闪点", "FP"]


def _load_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import BRICS
        from rdkit.Chem import BondType
    except ImportError as exc:
        raise ImportError(
            "RDKit is required for graph construction. Run with "
            "`qsar/.venv_smiles/bin/python`."
        ) from exc
    return Chem, BRICS, BondType


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise ValueError(f"input {x} not in allowable set {allowable_set}")
    return list(map(lambda s: x == s, allowable_set))


def get_implicit_valence(atom) -> int:
    Chem, _, _ = _load_rdkit()
    try:
        return int(atom.GetValence(Chem.ValenceType.IMPLICIT))
    except AttributeError:
        return int(atom.GetImplicitValence())


def get_atom_features(atom, stereo, explicit_hydrogen: bool = False, max_atomic_num: int = 100) -> list[int]:
    Chem, _, _ = _load_rdkit()
    atom_features = one_of_k_encoding_unk(atom.GetAtomicNum() - 1, list(range(max_atomic_num)))
    atom_features += one_of_k_encoding_unk(get_implicit_valence(atom), [0, 1, 2, 3])
    atom_features += one_of_k_encoding_unk(atom.GetNumRadicalElectrons(), [0, 1])
    atom_features += one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
    atom_features += one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, 0, 1])
    atom_features += one_of_k_encoding_unk(
        atom.GetHybridization(),
        [
            Chem.rdchem.HybridizationType.S,
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
        ],
    )
    atom_features += [atom.IsInRing()]
    if not explicit_hydrogen:
        atom_features += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

    try:
        atom_features += one_of_k_encoding_unk(stereo, ["R", "S"])
        atom_features += [atom.HasProp("_ChiralityPossible")]
    except Exception:
        atom_features += [False, False, atom.HasProp("_ChiralityPossible")]

    return [int(value) for value in atom_features]


def get_bond_features(bond) -> list[int]:
    _, _, BondType = _load_rdkit()
    bond_features = []
    bond_features += one_of_k_encoding_unk(
        bond.GetBondType(),
        [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC],
    )
    bond_features += [bond.GetIsAromatic()]
    bond_features += [bond.GetIsConjugated()]
    bond_features += one_of_k_encoding_unk(
        str(bond.GetStereo()),
        ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"],
    )
    return [int(value) for value in bond_features]


def canonicalize_fragment_smiles(fragment_smiles: list[str]) -> list[str]:
    Chem, _, _ = _load_rdkit()
    frags = [re.sub(r"\[[0-9]+\*\]", "*", frag) for frag in fragment_smiles]
    frags = [re.sub(r"\*", "[*:1]", frag) for frag in frags]
    normalized = []
    for frag in frags:
        mol = Chem.MolFromSmiles(frag)
        normalized.append(Chem.MolToSmiles(mol, isomericSmiles=False) if mol is not None else frag)
    return normalized


def build_atom_graph(mol) -> dict[str, Any]:
    Chem, _, _ = _load_rdkit()
    stereo = Chem.FindMolChiralCenters(mol)
    chiral_centers = [0] * mol.GetNumAtoms()
    for atom_idx, chirality in stereo:
        chiral_centers[atom_idx] = chirality

    node_features = []
    edge_index = []
    edge_features = []

    for atom_idx, atom in enumerate(mol.GetAtoms()):
        node_features.append(get_atom_features(atom, chiral_centers[atom_idx]))

    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        features = get_bond_features(bond)
        edge_index.append([begin, end])
        edge_features.append(features)
        edge_index.append([end, begin])
        edge_features.append(features)

    return {
        "num_nodes": mol.GetNumAtoms(),
        "node_feature_dim": len(node_features[0]) if node_features else 0,
        "edge_feature_dim": len(edge_features[0]) if edge_features else 0,
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
    }


def build_brics_fragment_graph(mol) -> dict[str, Any]:
    Chem, BRICS, _ = _load_rdkit()
    atom_count = mol.GetNumAtoms()
    atom_to_fragment = [0] * atom_count

    brics_bonds = [bond[0] for bond in list(BRICS.FindBRICSBonds(mol))]
    if not brics_bonds:
        smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
        return {
            "split_strategy": "brics",
            "num_fragments": 1,
            "fragment_smiles": [smiles],
            "fragment_atom_indices": [list(range(atom_count))],
            "atom_to_fragment": atom_to_fragment,
            "fragment_edge_index": [],
            "fragment_edge_features": [],
            "cut_bond_atom_pairs": [],
        }

    for atom_idx, atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(atom_idx + 1)

    cut_bond_indices = []
    cut_bond_atom_pairs = []
    cut_bond_features = []
    for begin, end in brics_bonds:
        bond = mol.GetBondBetweenAtoms(begin, end)
        if bond is None:
            continue
        cut_bond_indices.append(bond.GetIdx())
        cut_bond_atom_pairs.append([begin, end])
        cut_bond_features.append(get_bond_features(bond))

    fragmented = Chem.FragmentOnBonds(mol, cut_bond_indices)
    fragments = Chem.MolToSmiles(fragmented).split(".")
    frag_mols = [Chem.MolFromSmiles(frag) for frag in fragments]

    fragment_atom_indices = []
    for frag in frag_mols:
        atom_indices = []
        for atom in frag.GetAtoms():
            atom_map = atom.GetAtomMapNum()
            if atom_map > 0:
                atom_indices.append(atom_map - 1)
        fragment_atom_indices.append(sorted(set(atom_indices)))

    for frag_idx, atom_indices in enumerate(fragment_atom_indices):
        for atom_idx in atom_indices:
            atom_to_fragment[atom_idx] = frag_idx

    fragment_edge_index = []
    fragment_edge_features = []
    for (begin, end), feature in zip(cut_bond_atom_pairs, cut_bond_features):
        frag_begin = atom_to_fragment[begin]
        frag_end = atom_to_fragment[end]
        if frag_begin == frag_end:
            continue
        fragment_edge_index.append([frag_begin, frag_end])
        fragment_edge_features.append(feature)
        fragment_edge_index.append([frag_end, frag_begin])
        fragment_edge_features.append(feature)

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    return {
        "split_strategy": "brics",
        "num_fragments": len(fragment_atom_indices),
        "fragment_smiles": canonicalize_fragment_smiles(fragments),
        "fragment_atom_indices": fragment_atom_indices,
        "atom_to_fragment": atom_to_fragment,
        "fragment_edge_index": fragment_edge_index,
        "fragment_edge_features": fragment_edge_features,
        "cut_bond_atom_pairs": cut_bond_atom_pairs,
    }


def build_molecule_graph(
    smiles: str,
    molecule_id: str | None = None,
    molecule_name: str | None = None,
    target_value: float | None = None,
    split_strategy: str = "brics",
) -> dict[str, Any]:
    Chem, _, _ = _load_rdkit()
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return {
            "id": molecule_id,
            "name": molecule_name,
            "smiles": smiles,
            "valid": False,
            "target": target_value,
            "atom_graph": None,
            "fragment_graph": None,
        }

    if split_strategy != "brics":
        raise ValueError(f"Unsupported split_strategy: {split_strategy}")

    return {
        "id": molecule_id,
        "name": molecule_name,
        "smiles": Chem.MolToSmiles(mol, isomericSmiles=False),
        "valid": True,
        "target": target_value,
        "atom_graph": build_atom_graph(mol),
        "fragment_graph": build_brics_fragment_graph(mol),
    }


def prepare_training_frame(
    input_path: str | Path,
    smiles_column: str | None = None,
    target_column: str | None = None,
) -> tuple[pd.DataFrame, str, str | None]:
    df = read_table(input_path)
    smiles_column = smiles_column or pick_first_available(df, SMILES_CANDIDATES, "SMILES")

    if target_column is None:
        for candidate in TARGET_CANDIDATES:
            if candidate in df.columns:
                target_column = candidate
                break

    work_df = df.copy()
    work_df[smiles_column] = work_df[smiles_column].astype(str).str.strip()
    work_df = work_df[~work_df[smiles_column].isin(["", "nan", "NaN", "None"])]
    if target_column is not None:
        work_df[target_column] = work_df[target_column].map(extract_numeric)

    if ID_COL not in work_df.columns:
        work_df[ID_COL] = [f"mol_{i + 1:04d}" for i in range(len(work_df))]

    return work_df.reset_index(drop=True), smiles_column, target_column


def build_graph_dataset(
    input_path: str | Path,
    smiles_column: str | None = None,
    target_column: str | None = None,
    name_column: str | None = None,
    split_strategy: str = "brics",
) -> tuple[list[dict[str, Any]], dict[str, Any], pd.DataFrame]:
    frame, smiles_column, target_column = prepare_training_frame(
        input_path=input_path,
        smiles_column=smiles_column,
        target_column=target_column,
    )

    if name_column is None:
        for candidate in ["物质名称", "Name", "name"]:
            if candidate in frame.columns:
                name_column = candidate
                break

    graphs = []
    summary_rows = []
    for _, row in frame.iterrows():
        mol_id = str(row[ID_COL])
        mol_name = str(row[name_column]) if name_column and pd.notna(row[name_column]) else mol_id
        target_value = None
        if target_column is not None and pd.notna(row[target_column]):
            target_value = float(row[target_column])

        graph = build_molecule_graph(
            smiles=row[smiles_column],
            molecule_id=mol_id,
            molecule_name=mol_name,
            target_value=target_value,
            split_strategy=split_strategy,
        )
        graphs.append(graph)

        atom_graph = graph["atom_graph"] or {}
        fragment_graph = graph["fragment_graph"] or {}
        summary_rows.append(
            {
                "id": mol_id,
                "name": mol_name,
                "smiles": graph["smiles"],
                "valid": graph["valid"],
                "target": target_value,
                "num_atom_nodes": atom_graph.get("num_nodes", 0),
                "num_atom_edges": len(atom_graph.get("edge_index", [])),
                "num_fragments": fragment_graph.get("num_fragments", 0),
                "num_fragment_edges": len(fragment_graph.get("fragment_edge_index", [])),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    stats = {
        "input_path": str(input_path),
        "split_strategy": split_strategy,
        "smiles_column": smiles_column,
        "target_column": target_column,
        "name_column": name_column,
        "n_molecules": int(len(summary_df)),
        "n_valid": int(summary_df["valid"].sum()) if not summary_df.empty else 0,
        "avg_atom_nodes": float(summary_df["num_atom_nodes"].mean()) if not summary_df.empty else 0.0,
        "avg_fragments": float(summary_df["num_fragments"].mean()) if not summary_df.empty else 0.0,
    }
    return graphs, stats, summary_df


def save_graph_dataset(
    graphs: list[dict[str, Any]],
    stats: dict[str, Any],
    summary_df: pd.DataFrame,
    dataset_output: str | Path,
    stats_output: str | Path | None = None,
    summary_output: str | Path | None = None,
) -> None:
    dataset_output = Path(dataset_output)
    dataset_output.parent.mkdir(parents=True, exist_ok=True)
    with dataset_output.open("wb") as handle:
        pickle.dump({"graphs": graphs, "stats": stats}, handle)

    if stats_output is not None:
        stats_output = Path(stats_output)
        stats_output.parent.mkdir(parents=True, exist_ok=True)
        stats_output.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    if summary_output is not None:
        summary_output = Path(summary_output)
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_output, index=False)

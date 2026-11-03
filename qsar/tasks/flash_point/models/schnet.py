from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SchNetMoleculeGraph:
    smiles: str
    atomic_numbers: Tensor
    positions: Tensor
    edge_index: Tensor


@dataclass(frozen=True)
class BatchedSchNetGraph:
    atomic_numbers: Tensor
    positions: Tensor
    edge_index: Tensor
    batch_index: Tensor
    smiles: tuple[str, ...]

    @property
    def num_graphs(self) -> int:
        return len(self.smiles)

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> "BatchedSchNetGraph":
        return BatchedSchNetGraph(
            atomic_numbers=self.atomic_numbers.to(device, non_blocking=non_blocking),
            positions=self.positions.to(device, non_blocking=non_blocking),
            edge_index=self.edge_index.to(device, non_blocking=non_blocking),
            batch_index=self.batch_index.to(device, non_blocking=non_blocking),
            smiles=self.smiles,
        )


def _load_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise ImportError(
            "RDKit is required for SchNet conformer generation. Run with "
            "`qsar/.venv_smiles/bin/python`."
        ) from exc
    return Chem, AllChem


class SchNetFeaturizer:
    """Build a single-conformer 3D molecular graph from SMILES."""

    def __init__(
        self,
        *,
        cutoff: float = 5.0,
        add_hydrogens: bool = True,
        random_seed: int = 2026,
        optimize_steps: int = 200,
    ) -> None:
        self.cutoff = float(cutoff)
        self.add_hydrogens = bool(add_hydrogens)
        self.random_seed = int(random_seed)
        self.optimize_steps = int(optimize_steps)

    def __call__(self, smiles: str) -> SchNetMoleculeGraph:
        Chem, AllChem = _load_rdkit()
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
        if self.add_hydrogens:
            mol = Chem.AddHs(mol)

        params = AllChem.ETKDGv3()
        params.randomSeed = self.random_seed
        params.useRandomCoords = True
        status = AllChem.EmbedMolecule(mol, params)
        if status != 0:
            raise ValueError(f"Could not generate conformer for SMILES: {smiles}")
        if self.optimize_steps > 0:
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=self.optimize_steps)
            except Exception:
                pass

        conformer = mol.GetConformer()
        atomic_numbers = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)
        positions = torch.tensor(
            [
                [conformer.GetAtomPosition(i).x, conformer.GetAtomPosition(i).y, conformer.GetAtomPosition(i).z]
                for i in range(mol.GetNumAtoms())
            ],
            dtype=torch.float32,
        )
        return SchNetMoleculeGraph(
            smiles=canonical_smiles,
            atomic_numbers=atomic_numbers,
            positions=positions,
            edge_index=self._radius_graph(positions, self.cutoff),
        )

    @staticmethod
    def _radius_graph(positions: Tensor, cutoff: float) -> Tensor:
        num_atoms = positions.size(0)
        if num_atoms <= 1:
            return torch.empty((2, 0), dtype=torch.long)
        distance = torch.cdist(positions, positions)
        mask = (distance <= cutoff) & ~torch.eye(num_atoms, dtype=torch.bool)
        return mask.nonzero(as_tuple=False).t().contiguous().to(dtype=torch.long)


def collate_schnet_graphs(graphs: Sequence[SchNetMoleculeGraph]) -> BatchedSchNetGraph:
    if not graphs:
        raise ValueError("Cannot collate an empty SchNet graph batch.")

    atomic_numbers = []
    positions = []
    edge_indices = []
    batch_index = []
    smiles = []
    offset = 0
    for graph_idx, graph in enumerate(graphs):
        num_atoms = graph.atomic_numbers.size(0)
        atomic_numbers.append(graph.atomic_numbers)
        positions.append(graph.positions)
        edge_indices.append(graph.edge_index + offset)
        batch_index.append(torch.full((num_atoms,), graph_idx, dtype=torch.long))
        smiles.append(graph.smiles)
        offset += num_atoms

    return BatchedSchNetGraph(
        atomic_numbers=torch.cat(atomic_numbers, dim=0),
        positions=torch.cat(positions, dim=0),
        edge_index=torch.cat(edge_indices, dim=1),
        batch_index=torch.cat(batch_index, dim=0),
        smiles=tuple(smiles),
    )


class GaussianDistanceExpansion(nn.Module):
    def __init__(self, num_gaussians: int = 50, cutoff: float = 5.0) -> None:
        super().__init__()
        centers = torch.linspace(0.0, cutoff, num_gaussians)
        self.register_buffer("centers", centers)
        self.gamma = float(num_gaussians) / float(cutoff)

    def forward(self, distances: Tensor) -> Tensor:
        return torch.exp(-self.gamma * (distances.unsqueeze(-1) - self.centers).pow(2))


class SchNetInteraction(nn.Module):
    def __init__(self, hidden_dim: int, num_gaussians: int, dropout: float) -> None:
        super().__init__()
        self.filter_network = nn.Sequential(
            nn.Linear(num_gaussians, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, atom_states: Tensor, edge_index: Tensor, radial_features: Tensor, cutoff_values: Tensor) -> Tensor:
        if edge_index.numel() == 0:
            return atom_states
        source, target = edge_index
        filters = self.filter_network(radial_features) * cutoff_values.unsqueeze(-1)
        messages = atom_states[source] * filters
        aggregated = atom_states.new_zeros(atom_states.shape)
        aggregated.index_add_(0, target, messages)
        return self.norm(atom_states + self.update_network(aggregated))


class SchNetEncoder(nn.Module):
    """Reusable SchNet-style continuous-filter molecular encoder."""

    def __init__(
        self,
        *,
        max_atomic_number: int = 100,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_gaussians: int = 50,
        cutoff: float = 5.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cutoff = float(cutoff)
        self.hidden_dim = int(hidden_dim)
        self.embedding = nn.Embedding(max_atomic_number + 1, hidden_dim, padding_idx=0)
        self.distance_expansion = GaussianDistanceExpansion(num_gaussians=num_gaussians, cutoff=cutoff)
        self.interactions = nn.ModuleList(
            [
                SchNetInteraction(hidden_dim=hidden_dim, num_gaussians=num_gaussians, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, batch: BatchedSchNetGraph) -> Tensor:
        atom_states = self.embedding(batch.atomic_numbers.clamp_min(0))
        if batch.edge_index.numel():
            distance = torch.linalg.norm(
                batch.positions[batch.edge_index[0]] - batch.positions[batch.edge_index[1]],
                dim=-1,
            )
        else:
            distance = batch.positions.new_empty((0,))
        radial_features = self.distance_expansion(distance)
        cutoff_values = 0.5 * (torch.cos(torch.pi * distance / self.cutoff) + 1.0)
        cutoff_values = torch.where(distance <= self.cutoff, cutoff_values, torch.zeros_like(cutoff_values))

        for interaction in self.interactions:
            atom_states = interaction(atom_states, batch.edge_index, radial_features, cutoff_values)
        return atom_states


def schnet_mean_max_pool(atom_states: Tensor, batch_index: Tensor, num_graphs: int) -> Tensor:
    return torch.cat(
        [
            _scatter_mean(atom_states, batch_index, num_graphs),
            _scatter_max(atom_states, batch_index, num_graphs),
        ],
        dim=-1,
    )


class SchNetFlashPointRegressor(nn.Module):
    """SchNet 3D molecular encoder with a scalar flash-point readout."""

    def __init__(
        self,
        *,
        max_atomic_number: int = 100,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_gaussians: int = 50,
        cutoff: float = 5.0,
        readout_dim: int = 256,
        dropout: float = 0.1,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.encoder = SchNetEncoder(
            max_atomic_number=max_atomic_number,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, readout_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(readout_dim, readout_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(readout_dim // 2, output_dim),
        )

    def forward(self, batch: BatchedSchNetGraph) -> Tensor:
        atom_states = self.encoder(batch)
        pooled = schnet_mean_max_pool(atom_states, batch.batch_index, batch.num_graphs)
        return self.head(pooled)


def build_schnet_flash_point_gnn(**kwargs) -> SchNetFlashPointRegressor:
    return SchNetFlashPointRegressor(**kwargs)


def _scatter_mean(values: Tensor, batch_index: Tensor, num_graphs: int) -> Tensor:
    pooled = values.new_zeros((num_graphs, values.size(-1)))
    pooled.index_add_(0, batch_index, values)
    counts = values.new_zeros((num_graphs, 1))
    counts.index_add_(0, batch_index, torch.ones((values.size(0), 1), device=values.device))
    return pooled / counts.clamp_min(1.0)


def _scatter_max(values: Tensor, batch_index: Tensor, num_graphs: int) -> Tensor:
    pooled = values.new_full((num_graphs, values.size(-1)), -torch.inf)
    pooled.scatter_reduce_(
        0,
        batch_index[:, None].expand(-1, values.size(-1)),
        values,
        reduce="amax",
        include_self=True,
    )
    return pooled

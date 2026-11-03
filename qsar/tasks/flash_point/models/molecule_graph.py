from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

try:
    import torch
    from torch import Tensor
except ImportError as exc:  # pragma: no cover - exercised only in missing optional dependency envs.
    raise ImportError(
        "Molecule graph tensors require PyTorch. Install torch before using "
        "`qsar.tasks.flash_point.models.molecule_graph`."
    ) from exc

try:
    from qsar.core.graph_construction import build_molecule_graph
except ImportError:
    from ....core.graph_construction import build_molecule_graph


@dataclass(frozen=True)
class MoleculeGraphTensors:
    smiles: str
    node_features: Tensor
    edge_index: Tensor
    edge_features: Tensor


@dataclass(frozen=True)
class BatchedMoleculeGraph:
    node_features: Tensor
    edge_index: Tensor
    edge_features: Tensor
    batch_index: Tensor
    smiles: tuple[str, ...]

    @property
    def num_graphs(self) -> int:
        return len(self.smiles)

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> "BatchedMoleculeGraph":
        return BatchedMoleculeGraph(
            node_features=self.node_features.to(device, non_blocking=non_blocking),
            edge_index=self.edge_index.to(device, non_blocking=non_blocking),
            edge_features=self.edge_features.to(device, non_blocking=non_blocking),
            batch_index=self.batch_index.to(device, non_blocking=non_blocking),
            smiles=self.smiles,
        )


class MoleculeGraphFeaturizer:
    """Build atom-level graph tensors from SMILES using the local RDKit features."""

    def __init__(self, add_self_loops: bool = True) -> None:
        self.add_self_loops = add_self_loops

    def __call__(self, smiles: str) -> MoleculeGraphTensors:
        graph = build_molecule_graph(smiles=smiles, split_strategy="brics")
        if not graph["valid"] or graph["atom_graph"] is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        atom_graph = graph["atom_graph"]
        node_features = torch.tensor(atom_graph["node_features"], dtype=torch.float32)
        edge_features = torch.tensor(atom_graph["edge_features"], dtype=torch.float32)
        edge_index = torch.tensor(atom_graph["edge_index"], dtype=torch.long).t().contiguous()

        if edge_index.numel() == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_dim = int(atom_graph.get("edge_feature_dim", 0)) or 10
            edge_features = torch.empty((0, edge_dim), dtype=torch.float32)

        if self.add_self_loops:
            edge_index, edge_features = self._add_self_loops(
                edge_index=edge_index,
                edge_features=edge_features,
                num_nodes=node_features.size(0),
            )

        return MoleculeGraphTensors(
            smiles=str(graph["smiles"]),
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
        )

    @staticmethod
    def _add_self_loops(edge_index: Tensor, edge_features: Tensor, num_nodes: int) -> tuple[Tensor, Tensor]:
        self_loop_index = torch.arange(num_nodes, dtype=torch.long)
        self_loop_index = torch.stack([self_loop_index, self_loop_index], dim=0)
        edge_dim = edge_features.size(1) if edge_features.ndim == 2 and edge_features.numel() else 10
        self_loop_features = torch.zeros((num_nodes, edge_dim), dtype=torch.float32)
        return (
            torch.cat([edge_index, self_loop_index], dim=1),
            torch.cat([edge_features, self_loop_features], dim=0),
        )


def collate_molecule_graphs(graphs: Sequence[MoleculeGraphTensors]) -> BatchedMoleculeGraph:
    if not graphs:
        raise ValueError("Cannot collate an empty graph batch.")

    node_features = []
    edge_indices = []
    edge_features = []
    batch_index = []
    smiles = []
    node_offset = 0

    for graph_idx, graph in enumerate(graphs):
        num_nodes = graph.node_features.size(0)
        node_features.append(graph.node_features)
        edge_indices.append(graph.edge_index + node_offset)
        edge_features.append(graph.edge_features)
        batch_index.append(torch.full((num_nodes,), graph_idx, dtype=torch.long))
        smiles.append(graph.smiles)
        node_offset += num_nodes

    return BatchedMoleculeGraph(
        node_features=torch.cat(node_features, dim=0),
        edge_index=torch.cat(edge_indices, dim=1),
        edge_features=torch.cat(edge_features, dim=0),
        batch_index=torch.cat(batch_index, dim=0),
        smiles=tuple(smiles),
    )


def scatter_mean(values: Tensor, batch_index: Tensor, num_graphs: int) -> Tensor:
    pooled = values.new_zeros((num_graphs, values.size(-1)))
    pooled.index_add_(0, batch_index, values)
    counts = values.new_zeros((num_graphs, 1))
    counts.index_add_(0, batch_index, torch.ones((values.size(0), 1), device=values.device))
    return pooled / counts.clamp_min(1.0)


def scatter_max(values: Tensor, batch_index: Tensor, num_graphs: int) -> Tensor:
    pooled = values.new_full((num_graphs, values.size(-1)), -torch.inf)
    pooled.scatter_reduce_(
        0,
        batch_index[:, None].expand(-1, values.size(-1)),
        values,
        reduce="amax",
        include_self=True,
    )
    return pooled

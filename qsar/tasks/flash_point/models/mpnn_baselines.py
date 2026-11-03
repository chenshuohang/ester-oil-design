from __future__ import annotations

from collections.abc import Sequence

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover - exercised only in missing optional dependency envs.
    raise ImportError(
        "The MPNN baselines require PyTorch. Install torch before using "
        "`qsar.tasks.flash_point.models.mpnn_baselines`."
    ) from exc

try:
    from qsar.tasks.flash_point.models.molecule_graph import BatchedMoleculeGraph
except ImportError:
    from .molecule_graph import BatchedMoleculeGraph


class DirectedMPNNRegressor(nn.Module):
    """Directed bond-message MPNN baseline inspired by Chemprop D-MPNN.

    Messages live on directed bonds. At each step, a bond receives messages from
    bonds entering its source atom, excluding the immediate reverse bond.
    """

    def __init__(
        self,
        node_feature_dim: int = 130,
        edge_feature_dim: int = 10,
        hidden_dim: int = 128,
        num_layers: int = 4,
        readout_dim: int = 256,
        output_dim: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")

        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.edge_encoder = nn.Sequential(
            nn.Linear(node_feature_dim + edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.message_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.atom_update = nn.Sequential(
            nn.Linear(node_feature_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, readout_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(readout_dim, readout_dim // 2),
            nn.ReLU(),
        )
        self.head = nn.Linear(readout_dim // 2, output_dim)

    def forward(self, batch: BatchedMoleculeGraph) -> Tensor:
        _validate_feature_dims(batch, self.node_feature_dim, self.edge_feature_dim)

        source, target = batch.edge_index
        reverse_edge = _reverse_edge_index(batch.edge_index)
        edge_states = self.edge_encoder(torch.cat([batch.node_features[source], batch.edge_features], dim=-1))
        initial_edge_states = edge_states

        for _ in range(self.num_layers - 1):
            incoming = edge_states.new_zeros(edge_states.shape)
            incoming_to_source = edge_states.new_zeros((batch.node_features.size(0), edge_states.size(-1)))
            incoming_to_source.index_add_(0, target, edge_states)
            incoming = incoming_to_source[source]

            has_reverse = reverse_edge >= 0
            reverse_messages = edge_states.new_zeros(edge_states.shape)
            reverse_messages[has_reverse] = edge_states[reverse_edge[has_reverse]]
            incoming = incoming - reverse_messages

            edge_states = initial_edge_states + self.message_layer(incoming)

        atom_messages = batch.node_features.new_zeros((batch.node_features.size(0), edge_states.size(-1)))
        atom_messages.index_add_(0, target, edge_states)
        node_states = self.atom_update(torch.cat([batch.node_features, atom_messages], dim=-1))

        mean_pool = _scatter_mean(node_states, batch.batch_index, batch.num_graphs)
        max_pool = _scatter_max(node_states, batch.batch_index, batch.num_graphs)
        graph_states = self.readout(torch.cat([mean_pool, max_pool], dim=-1))
        return self.head(graph_states)


class CommunicativeMPNNLayer(nn.Module):
    """One atom-bond communication block for Communicative MPNN."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.edge_message = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_update = nn.GRUCell(hidden_dim, hidden_dim)
        self.node_message = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_update = nn.GRUCell(hidden_dim, hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_states: Tensor, edge_states: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        source, target = edge_index
        edge_messages = self.edge_message(
            torch.cat([node_states[source], node_states[target], edge_states], dim=-1)
        )
        edge_states = self.edge_norm(self.edge_update(edge_messages, edge_states) + edge_states)

        node_messages = self.node_message(torch.cat([node_states[source], edge_states], dim=-1))
        aggregated = node_states.new_zeros(node_states.shape)
        aggregated.index_add_(0, target, node_messages)

        degree = node_states.new_zeros((node_states.size(0), 1))
        degree.index_add_(0, target, torch.ones((target.size(0), 1), device=node_states.device))
        aggregated = aggregated / degree.clamp_min(1.0)

        node_states = self.node_norm(self.node_update(aggregated, node_states) + node_states)
        return node_states, edge_states


class CommunicativeMPNNRegressor(nn.Module):
    """Atom-bond co-evolution MPNN baseline.

    Both atoms and directed bonds keep hidden states. Each layer first lets atom
    pairs update the bond state, then lets updated bonds communicate back to the
    destination atom.
    """

    def __init__(
        self,
        node_feature_dim: int = 130,
        edge_feature_dim: int = 10,
        hidden_dim: int = 128,
        num_layers: int = 4,
        readout_dim: int = 256,
        output_dim: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")

        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.output_dim = output_dim

        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList(
            [CommunicativeMPNNLayer(hidden_dim=hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, readout_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(readout_dim, readout_dim // 2),
            nn.ReLU(),
        )
        self.head = nn.Linear(readout_dim // 2, output_dim)

    def forward(self, batch: BatchedMoleculeGraph) -> Tensor:
        _validate_feature_dims(batch, self.node_feature_dim, self.edge_feature_dim)

        node_states = self.node_encoder(batch.node_features)
        edge_states = self.edge_encoder(batch.edge_features)
        for layer in self.layers:
            node_states, edge_states = layer(node_states, edge_states, batch.edge_index)

        mean_pool = _scatter_mean(node_states, batch.batch_index, batch.num_graphs)
        max_pool = _scatter_max(node_states, batch.batch_index, batch.num_graphs)
        graph_states = self.readout(torch.cat([mean_pool, max_pool], dim=-1))
        return self.head(graph_states)


# Alias the user's requested spelling so imports remain forgiving.
CommunitativeMPNNRegressor = CommunicativeMPNNRegressor


def _validate_feature_dims(batch: BatchedMoleculeGraph, node_feature_dim: int, edge_feature_dim: int) -> None:
    if batch.node_features.size(-1) != node_feature_dim:
        raise ValueError(f"Expected node feature dim {node_feature_dim}, got {batch.node_features.size(-1)}.")
    if batch.edge_features.size(-1) != edge_feature_dim:
        raise ValueError(f"Expected edge feature dim {edge_feature_dim}, got {batch.edge_features.size(-1)}.")


def _reverse_edge_index(edge_index: Tensor) -> Tensor:
    lookup = {
        (int(source), int(target)): idx
        for idx, (source, target) in enumerate(edge_index.t().detach().cpu().tolist())
    }
    reverse = [lookup.get((int(target), int(source)), -1) for source, target in edge_index.t().detach().cpu().tolist()]
    return torch.tensor(reverse, dtype=torch.long, device=edge_index.device)


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


def build_mpnn_baseline(
    name: str,
    *,
    node_feature_dim: int = 130,
    edge_feature_dim: int = 10,
    hidden_dim: int = 128,
    num_layers: int = 4,
    readout_dim: int = 256,
    output_dim: int = 3,
    dropout: float = 0.1,
) -> nn.Module:
    normalized_name = name.strip().lower().replace("_", "-")
    config = {
        "node_feature_dim": node_feature_dim,
        "edge_feature_dim": edge_feature_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "readout_dim": readout_dim,
        "output_dim": output_dim,
        "dropout": dropout,
    }
    if normalized_name in {"directed", "d-mpnn", "directed-mpnn"}:
        return DirectedMPNNRegressor(**config)
    if normalized_name in {"communicative", "communitative", "c-mpnn", "communicative-mpnn"}:
        return CommunicativeMPNNRegressor(**config)
    raise ValueError(f"Unknown MPNN baseline: {name!r}. Expected 'directed' or 'communicative'.")


def mpnn_baseline_names() -> Sequence[str]:
    return ("directed", "communicative")

from __future__ import annotations

import torch
from torch import Tensor, nn

from qsar.tasks.flash_point.models.mpnn_baselines import build_mpnn_baseline
from qsar.tasks.flash_point.models.molecule_graph import BatchedMoleculeGraph


class ConditionalViscosityMPNN(nn.Module):
    """MPNN molecular encoder with temperature/pressure conditioning."""

    def __init__(
        self,
        *,
        backbone_name: str = "communicative",
        node_feature_dim: int = 130,
        edge_feature_dim: int = 10,
        hidden_dim: int = 128,
        num_layers: int = 4,
        readout_dim: int = 256,
        molecule_dim: int | None = None,
        condition_dim: int = 64,
        dropout: float = 0.1,
        reference_temperature_K: float = 298.15,
        reference_pressure_kPa: float = 101.325,
        temperature_scale_K: float = 100.0,
        pressure_scale_kPa: float = 100000.0,
    ) -> None:
        super().__init__()
        if condition_dim < 1:
            raise ValueError("condition_dim must be >= 1.")

        self.backbone_name = backbone_name
        self.reference_temperature_K = float(reference_temperature_K)
        self.reference_pressure_kPa = float(reference_pressure_kPa)
        self.temperature_scale_K = float(temperature_scale_K)
        self.pressure_scale_kPa = float(pressure_scale_kPa)

        resolved_molecule_dim = molecule_dim or readout_dim // 2
        if resolved_molecule_dim < 1:
            raise ValueError("molecule_dim must be >= 1.")
        self.molecule_dim = int(resolved_molecule_dim)
        self.condition_dim = int(condition_dim)

        self.backbone = build_mpnn_baseline(
            backbone_name,
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            readout_dim=readout_dim,
            output_dim=self.molecule_dim,
            dropout=dropout,
        )
        self.condition_encoder = nn.Sequential(
            nn.Linear(2, condition_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(condition_dim, condition_dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(self.molecule_dim + condition_dim, readout_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(readout_dim // 2, 1),
        )

    def encode_conditions(self, temperature_K: Tensor, pressure_kPa: Tensor, dtype: torch.dtype) -> Tensor:
        conditions = torch.stack(
            [
                (temperature_K.reshape(-1) - self.reference_temperature_K) / self.temperature_scale_K,
                (pressure_kPa.reshape(-1) - self.reference_pressure_kPa) / self.pressure_scale_kPa,
            ],
            dim=-1,
        )
        return self.condition_encoder(conditions.to(dtype=dtype))

    def forward(self, batch: BatchedMoleculeGraph, temperature_K: Tensor, pressure_kPa: Tensor) -> Tensor:
        molecule_state = self.backbone(batch)
        condition_state = self.encode_conditions(temperature_K, pressure_kPa, dtype=molecule_state.dtype)
        return self.head(torch.cat([molecule_state, condition_state], dim=-1)).squeeze(-1)


def build_conditional_viscosity_mpnn(**kwargs) -> ConditionalViscosityMPNN:
    return ConditionalViscosityMPNN(**kwargs)

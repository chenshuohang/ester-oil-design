from __future__ import annotations

import torch
from torch import Tensor, nn

from qsar.tasks.flash_point.models.mpnn_baselines import build_mpnn_baseline
from qsar.tasks.flash_point.models.molecule_graph import BatchedMoleculeGraph


class TemperatureAwareViscosityMPNN(nn.Module):
    """Predict molecular viscosity surfaces: log(eta) = A(mol) + B_T(mol)T + B_P(mol)P."""

    def __init__(
        self,
        *,
        backbone_name: str = "directed",
        node_feature_dim: int = 130,
        edge_feature_dim: int = 10,
        hidden_dim: int = 128,
        num_layers: int = 4,
        readout_dim: int = 256,
        dropout: float = 0.1,
        reference_temperature_K: float = 298.15,
        reference_pressure_kPa: float = 101.325,
        temperature_scale_K: float = 100.0,
        pressure_scale_kPa: float = 100000.0,
        enforce_negative_slope: bool = False,
        enforce_positive_pressure_slope: bool = False,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.reference_temperature_K = float(reference_temperature_K)
        self.reference_pressure_kPa = float(reference_pressure_kPa)
        self.temperature_scale_K = float(temperature_scale_K)
        self.pressure_scale_kPa = float(pressure_scale_kPa)
        self.enforce_negative_slope = bool(enforce_negative_slope)
        self.enforce_positive_pressure_slope = bool(enforce_positive_pressure_slope)
        self.backbone = build_mpnn_baseline(
            backbone_name,
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            readout_dim=readout_dim,
            output_dim=3,
            dropout=dropout,
        )

    def molecular_parameters(self, batch: BatchedMoleculeGraph) -> tuple[Tensor, Tensor, Tensor]:
        alpha, raw_temperature_slope, raw_pressure_slope = self.backbone(batch).unbind(dim=-1)
        scaled_temperature_slope = (
            -torch.nn.functional.softplus(raw_temperature_slope)
            if self.enforce_negative_slope
            else raw_temperature_slope
        )
        scaled_pressure_slope = (
            torch.nn.functional.softplus(raw_pressure_slope)
            if self.enforce_positive_pressure_slope
            else raw_pressure_slope
        )
        temperature_slope = scaled_temperature_slope / self.temperature_scale_K
        pressure_slope = scaled_pressure_slope / self.pressure_scale_kPa
        intercept = (
            alpha
            - temperature_slope * self.reference_temperature_K
            - pressure_slope * self.reference_pressure_kPa
        )
        return intercept, temperature_slope, pressure_slope

    def forward(self, batch: BatchedMoleculeGraph, temperature_K: Tensor, pressure_kPa: Tensor) -> Tensor:
        alpha, raw_temperature_slope, raw_pressure_slope = self.backbone(batch).unbind(dim=-1)
        temperature_slope = (
            -torch.nn.functional.softplus(raw_temperature_slope)
            if self.enforce_negative_slope
            else raw_temperature_slope
        )
        pressure_slope = (
            torch.nn.functional.softplus(raw_pressure_slope)
            if self.enforce_positive_pressure_slope
            else raw_pressure_slope
        )
        return (
            alpha
            + temperature_slope * (temperature_K.reshape(-1) - self.reference_temperature_K) / self.temperature_scale_K
            + pressure_slope * (pressure_kPa.reshape(-1) - self.reference_pressure_kPa) / self.pressure_scale_kPa
        )


def build_temperature_viscosity_mpnn(**kwargs) -> TemperatureAwareViscosityMPNN:
    return TemperatureAwareViscosityMPNN(**kwargs)

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from qsar.tasks.flash_point.models.schnet import (
    BatchedSchNetGraph,
    SchNetEncoder,
    SchNetFeaturizer,
    collate_schnet_graphs,
    schnet_mean_max_pool,
)


class ConditionalSchNetViscosityRegressor(nn.Module):
    """SchNet 3D molecular encoder with temperature/pressure conditioning."""

    def __init__(
        self,
        *,
        max_atomic_number: int = 100,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_gaussians: int = 50,
        cutoff: float = 5.0,
        readout_dim: int = 256,
        condition_dim: int = 64,
        dropout: float = 0.1,
        reference_temperature_K: float = 298.15,
        reference_pressure_kPa: float = 101.325,
        temperature_scale_K: float = 100.0,
        pressure_scale_kPa: float = 100000.0,
    ) -> None:
        super().__init__()
        self.reference_temperature_K = float(reference_temperature_K)
        self.reference_pressure_kPa = float(reference_pressure_kPa)
        self.temperature_scale_K = float(temperature_scale_K)
        self.pressure_scale_kPa = float(pressure_scale_kPa)

        self.encoder = SchNetEncoder(
            max_atomic_number=max_atomic_number,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            dropout=dropout,
        )
        self.graph_readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, readout_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(readout_dim, readout_dim // 2),
            nn.SiLU(),
        )
        self.condition_encoder = nn.Sequential(
            nn.Linear(2, condition_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(condition_dim, condition_dim),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(readout_dim // 2 + condition_dim, readout_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(readout_dim // 2, 1),
        )

    def forward(self, batch: BatchedSchNetGraph, temperature_K: Tensor, pressure_kPa: Tensor) -> Tensor:
        atom_states = self.encoder(batch)
        pooled = schnet_mean_max_pool(atom_states, batch.batch_index, batch.num_graphs)
        graph_state = self.graph_readout(pooled)

        conditions = torch.stack(
            [
                (temperature_K.reshape(-1) - self.reference_temperature_K) / self.temperature_scale_K,
                (pressure_kPa.reshape(-1) - self.reference_pressure_kPa) / self.pressure_scale_kPa,
            ],
            dim=-1,
        )
        condition_state = self.condition_encoder(conditions.to(dtype=graph_state.dtype))
        return self.head(torch.cat([graph_state, condition_state], dim=-1)).squeeze(-1)


def build_schnet_viscosity_gnn(**kwargs) -> ConditionalSchNetViscosityRegressor:
    return ConditionalSchNetViscosityRegressor(**kwargs)


class DynamicViscositySchNetPredictor:
    def __init__(
        self,
        model: ConditionalSchNetViscosityRegressor,
        checkpoint_path: str | Path | None = None,
        device: str | torch.device | None = None,
        target_mean: float | Tensor | None = None,
        target_std: float | Tensor | None = None,
        featurizer_kwargs: dict | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.featurizer = SchNetFeaturizer(**(featurizer_kwargs or {}))
        self.target_mean = self._as_scalar_tensor(target_mean)
        self.target_std = self._as_scalar_tensor(target_std)
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        self.model.eval()

    @staticmethod
    def _as_scalar_tensor(value: float | Tensor | None) -> Tensor | None:
        if value is None:
            return None
        return torch.as_tensor(value, dtype=torch.float32).reshape(())

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
    ) -> "DynamicViscositySchNetPredictor":
        checkpoint = torch.load(checkpoint_path, map_location=device or "cpu", weights_only=False)
        model = ConditionalSchNetViscosityRegressor(**checkpoint.get("model_config", {}))
        return cls(
            model=model,
            checkpoint_path=checkpoint_path,
            device=device,
            target_mean=checkpoint.get("target_mean"),
            target_std=checkpoint.get("target_std"),
            featurizer_kwargs=checkpoint.get("featurizer_config", {}),
        )

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        self.target_mean = self._as_scalar_tensor(checkpoint.get("target_mean", self.target_mean))
        self.target_std = self._as_scalar_tensor(checkpoint.get("target_std", self.target_std))
        self.model.eval()

    def predict(self, smiles: str, temperature_K: float, pressure_kPa: float = 101.325) -> float:
        graph = collate_schnet_graphs([self.featurizer(smiles)]).to(self.device)
        temperature = torch.tensor([temperature_K], dtype=torch.float32, device=self.device)
        pressure = torch.tensor([pressure_kPa], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            scaled = self.model(graph, temperature, pressure)
            if self.target_mean is not None and self.target_std is not None:
                scaled = scaled * self.target_std.to(self.device) + self.target_mean.to(self.device)
            return float(torch.exp(scaled).item())

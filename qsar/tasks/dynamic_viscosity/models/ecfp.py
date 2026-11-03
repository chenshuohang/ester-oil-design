from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn

from qsar.tasks.flash_point.models.ecfp import (
    BatchedECFPFingerprint,
    ECFPFeaturizer,
    collate_ecfp_fingerprints,
)


def _load_torch_checkpoint(path: str | Path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class ConditionalECFPViscosityRegressor(nn.Module):
    """ECFP fingerprint MLP with temperature/pressure conditioning."""

    def __init__(
        self,
        *,
        input_dim: int = 2048,
        hidden_dim: int = 512,
        num_layers: int = 3,
        readout_dim: int = 256,
        condition_dim: int = 64,
        dropout: float = 0.1,
        reference_temperature_K: float = 298.15,
        reference_pressure_kPa: float = 101.325,
        temperature_scale_K: float = 100.0,
        pressure_scale_kPa: float = 100000.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0.")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")
        if condition_dim < 1:
            raise ValueError("condition_dim must be >= 1.")

        self.reference_temperature_K = float(reference_temperature_K)
        self.reference_pressure_kPa = float(reference_pressure_kPa)
        self.temperature_scale_K = float(temperature_scale_K)
        self.pressure_scale_kPa = float(pressure_scale_kPa)

        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim

        self.fingerprint_encoder = nn.Sequential(*layers)
        self.molecule_readout = nn.Sequential(
            nn.Linear(current_dim, readout_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(readout_dim, readout_dim // 2),
            nn.ReLU(),
        )
        self.condition_encoder = nn.Sequential(
            nn.Linear(2, condition_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(condition_dim, condition_dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(readout_dim // 2 + condition_dim, readout_dim // 2),
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

    def forward(
        self,
        batch: BatchedECFPFingerprint | Tensor,
        temperature_K: Tensor,
        pressure_kPa: Tensor,
    ) -> Tensor:
        fingerprints = batch.fingerprints if isinstance(batch, BatchedECFPFingerprint) else batch
        molecule_state = self.molecule_readout(self.fingerprint_encoder(fingerprints))
        condition_state = self.encode_conditions(temperature_K, pressure_kPa, dtype=molecule_state.dtype)
        return self.head(torch.cat([molecule_state, condition_state], dim=-1)).squeeze(-1)


def build_ecfp_viscosity_model(**kwargs) -> ConditionalECFPViscosityRegressor:
    return ConditionalECFPViscosityRegressor(**kwargs)


class ECFPViscosityPredictor:
    """Convenience wrapper for checkpointed ECFP dynamic-viscosity models."""

    def __init__(
        self,
        model: ConditionalECFPViscosityRegressor,
        *,
        featurizer: ECFPFeaturizer | None = None,
        target_mean: Tensor | float | None = None,
        target_std: Tensor | float | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.featurizer = featurizer or ECFPFeaturizer()
        self.target_mean = self._as_scalar_tensor(target_mean)
        self.target_std = self._as_scalar_tensor(target_std)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
    ) -> "ECFPViscosityPredictor":
        checkpoint = _load_torch_checkpoint(checkpoint_path, map_location=device or "cpu")
        model = build_ecfp_viscosity_model(**checkpoint["model_config"])
        predictor = cls(
            model,
            featurizer=ECFPFeaturizer(**checkpoint.get("featurizer_config", {})),
            target_mean=checkpoint.get("target_mean"),
            target_std=checkpoint.get("target_std"),
            device=device,
        )
        predictor.model.load_state_dict(checkpoint["model_state_dict"])
        predictor.model.eval()
        return predictor

    @torch.no_grad()
    def predict_smiles(
        self,
        smiles: str | Sequence[str],
        temperature_K: float | Sequence[float],
        pressure_kPa: float | Sequence[float] = 101.325,
        batch_size: int = 64,
        return_tensor: bool = False,
    ) -> list[float] | Tensor:
        smiles_list = [smiles] if isinstance(smiles, str) else list(smiles)
        temperatures = self._expand_condition(temperature_K, len(smiles_list), "temperature_K")
        pressures = self._expand_condition(pressure_kPa, len(smiles_list), "pressure_kPa")
        outputs = []

        for start in range(0, len(smiles_list), batch_size):
            end = start + batch_size
            fingerprints = [self.featurizer(item) for item in smiles_list[start:end]]
            batch = collate_ecfp_fingerprints(fingerprints).to(self.device)
            temperature = torch.tensor(temperatures[start:end], dtype=torch.float32, device=self.device)
            pressure = torch.tensor(pressures[start:end], dtype=torch.float32, device=self.device)
            prediction = self.model(batch, temperature, pressure)
            outputs.append(self._inverse_transform(prediction).cpu())

        predictions = torch.cat(outputs, dim=0) if outputs else torch.empty(0)
        if return_tensor:
            return predictions
        return [float(value) for value in predictions.tolist()]

    def _inverse_transform(self, prediction: Tensor) -> Tensor:
        if self.target_mean is None or self.target_std is None:
            return prediction
        return prediction * self.target_std.to(prediction.device) + self.target_mean.to(prediction.device)

    def _as_scalar_tensor(self, value: float | Tensor | None) -> Tensor | None:
        if value is None:
            return None
        return torch.as_tensor(value, dtype=torch.float32, device=self.device).reshape(())

    @staticmethod
    def _expand_condition(value: float | Sequence[float], length: int, name: str) -> list[float]:
        if isinstance(value, (int, float)):
            return [float(value)] * length
        values = [float(item) for item in value]
        if len(values) != length:
            raise ValueError(f"{name} must be a scalar or have one value per SMILES.")
        return values

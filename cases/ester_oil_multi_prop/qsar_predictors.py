from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qsar.tasks.flash_point.models.ecfp import (  # noqa: E402
    ECFPFeaturizer,
    build_ecfp_flash_point_model,
    collate_ecfp_fingerprints,
)
from qsar.tasks.flash_point.models.molecule_graph import (  # noqa: E402
    MoleculeGraphFeaturizer,
    collate_molecule_graphs,
)
from qsar.tasks.flash_point.models.mpnn_baselines import build_mpnn_baseline  # noqa: E402
from qsar.tasks.flash_point.models.schnet import (  # noqa: E402
    SchNetFeaturizer,
    build_schnet_flash_point_gnn,
    collate_schnet_graphs,
)
from qsar.tasks.dynamic_viscosity.models.ecfp import build_ecfp_viscosity_model  # noqa: E402
from qsar.tasks.dynamic_viscosity.models.schnet import build_schnet_viscosity_gnn  # noqa: E402
from qsar.tasks.dynamic_viscosity.models.temperature_mpnn import build_temperature_viscosity_mpnn  # noqa: E402
from qsar.tasks.dynamic_viscosity.models.conditional_mpnn import build_conditional_viscosity_mpnn  # noqa: E402


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_torch_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(resolve_project_path(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(resolve_project_path(path), map_location=device)


class QsarObjective:
    """Batch predictor wrapper with a small SMILES cache for evolution loops."""

    def __init__(self, spec: dict[str, Any], *, device: str | torch.device | None = None) -> None:
        self.spec = dict(spec)
        self.name = str(spec["name"])
        self.kind = str(spec.get("kind", "auto")).lower()
        self.direction = str(spec.get("direction", "minimize")).lower()
        self.target_key = spec.get("target_key")
        self.invalid_value = float(spec.get("invalid_value", math.nan))
        self.temperature_K = float(spec.get("temperature_K", 298.15))
        self.pressure_kPa = float(spec.get("pressure_kPa", 101.325))
        self.output_transform = str(spec.get("output_transform", "identity")).lower()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint_path = resolve_project_path(spec["checkpoint"])
        self.checkpoint = load_torch_checkpoint(self.checkpoint_path, self.device)
        self.model_name = str(self.checkpoint.get("model_name", "")).lower()
        self.target_columns = tuple(
            self.checkpoint.get("target_columns")
            or self.checkpoint.get("property_names")
            or ()
        )
        self.model, self.featurizer, self.featurizer_type = self._build_model_and_featurizer()
        self.model.to(self.device)
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.eval()
        self.target_mean = self._as_tensor(self.checkpoint.get("target_mean"))
        self.target_std = self._as_tensor(self.checkpoint.get("target_std"))
        self.cache: dict[str, float] = {}

    def _as_tensor(self, value: Any) -> torch.Tensor | None:
        if value is None:
            return None
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def _build_model_and_featurizer(self):
        config = self.checkpoint.get("model_config", {})
        if self.kind == "viscosity" or "viscosity" in self.model_name:
            return self._build_viscosity_model(config)
        return self._build_scalar_model(config)

    def _build_scalar_model(self, config: dict[str, Any]):
        if "ecfp" in self.model_name:
            return (
                build_ecfp_flash_point_model(**config),
                ECFPFeaturizer(**self.checkpoint.get("featurizer_config", {})),
                "ecfp",
            )
        if "schnet" in self.model_name:
            return (
                build_schnet_flash_point_gnn(**config),
                SchNetFeaturizer(**self.checkpoint.get("featurizer_config", {})),
                "schnet",
            )

        normalized = "communicative" if "communicative" in self.model_name else "directed"
        return (
            build_mpnn_baseline(normalized, **config),
            MoleculeGraphFeaturizer(add_self_loops=normalized != "directed"),
            "molecule_graph",
        )

    def _build_viscosity_model(self, config: dict[str, Any]):
        if "ecfp" in self.model_name:
            return (
                build_ecfp_viscosity_model(**config),
                ECFPFeaturizer(**self.checkpoint.get("featurizer_config", {})),
                "conditioned_ecfp",
            )
        if "schnet" in self.model_name:
            return (
                build_schnet_viscosity_gnn(**config),
                SchNetFeaturizer(**self.checkpoint.get("featurizer_config", {})),
                "conditioned_schnet",
            )

        backbone_name = str(config.get("backbone_name", "communicative")).lower()
        add_self_loops = backbone_name != "directed"

        # Some saved checkpoints have generic model names, so prefer the
        # serialized architecture config over filename/model-name aliases.
        is_temperature_aware = (
            "temperature_aware" in self.model_name
            or "enforce_negative_slope" in config
            or "enforce_positive_pressure_slope" in config
        )
        if is_temperature_aware:
            return (
                build_temperature_viscosity_mpnn(**config),
                MoleculeGraphFeaturizer(add_self_loops=add_self_loops),
                "conditioned_molecule_graph",
            )

        return (
            build_conditional_viscosity_mpnn(**config),
            MoleculeGraphFeaturizer(add_self_loops=add_self_loops),
            "conditioned_molecule_graph",
        )

    def predict(self, smiles: Sequence[str], *, batch_size: int = 64) -> np.ndarray:
        smiles_list = [str(item) for item in smiles]
        missing = [item for item in dict.fromkeys(smiles_list) if item not in self.cache]
        for start in range(0, len(missing), batch_size):
            self._predict_uncached_batch(missing[start : start + batch_size])
        return np.asarray([self.cache.get(item, self.invalid_value) for item in smiles_list], dtype=np.float32)

    @torch.no_grad()
    def _predict_uncached_batch(self, smiles: Sequence[str]) -> None:
        graphs = []
        valid_smiles = []
        for item in smiles:
            try:
                graphs.append(self.featurizer(item))
                valid_smiles.append(item)
            except (ImportError, ValueError):
                self.cache[item] = self.invalid_value

        if not graphs:
            return

        if self.featurizer_type.endswith("ecfp"):
            batch = collate_ecfp_fingerprints(graphs).to(self.device)
            prediction = self._call_model(batch, len(graphs))
        elif self.featurizer_type.endswith("schnet"):
            batch = collate_schnet_graphs(graphs).to(self.device)
            prediction = self._call_model(batch, len(graphs))
        else:
            batch = collate_molecule_graphs(graphs).to(self.device)
            prediction = self._call_model(batch, len(graphs))

        prediction = self._inverse_transform(prediction).detach().cpu().reshape(len(graphs), -1)
        for item, row in zip(valid_smiles, prediction.tolist()):
            self.cache[item] = self._extract_value(row)

    def _call_model(self, batch: Any, n_items: int) -> torch.Tensor:
        if self.featurizer_type.startswith("conditioned"):
            temperature = torch.full((n_items,), self.temperature_K, dtype=torch.float32, device=self.device)
            pressure = torch.full((n_items,), self.pressure_kPa, dtype=torch.float32, device=self.device)
            return self.model(batch, temperature, pressure)
        return self.model(batch)

    def _inverse_transform(self, prediction: torch.Tensor) -> torch.Tensor:
        if self.target_mean is not None and self.target_std is not None:
            prediction = prediction * self.target_std.to(prediction.device) + self.target_mean.to(prediction.device)
        if self.output_transform == "exp":
            prediction = torch.exp(prediction)
        return prediction

    def _extract_value(self, values: Sequence[float]) -> float:
        if self.target_key and self.target_columns:
            try:
                idx = [str(item) for item in self.target_columns].index(str(self.target_key))
            except ValueError:
                idx = 0
        else:
            idx = 0
        return float(values[idx])

    def score(self, values: np.ndarray) -> np.ndarray:
        values = values.astype(np.float32, copy=True)
        if self.direction == "maximize":
            return -values
        if self.direction == "target":
            target = float(self.spec["target"])
            return np.abs(values - target)
        return values


class MultiObjectiveQsarScorer:
    def __init__(self, specs: Sequence[dict[str, Any]], *, device: str | torch.device | None = None, batch_size: int = 64) -> None:
        self.objectives = [QsarObjective(spec, device=device) for spec in specs]
        self.batch_size = int(batch_size)

    def __call__(self, smiles: Sequence[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        raw: dict[str, np.ndarray] = {}
        scores = []
        for objective in self.objectives:
            values = objective.predict(smiles, batch_size=self.batch_size)
            raw[objective.name] = values
            score = objective.score(values)
            score[~np.isfinite(score)] = 1e6
            scores.append(score)
        return np.stack(scores, axis=1).astype(np.float32), raw

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover - exercised only in missing optional dependency envs.
    raise ImportError(
        "The ECFP baseline requires PyTorch. Install torch before using "
        "`qsar.tasks.flash_point.models.ecfp`."
    ) from exc


def _load_rdkit():
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError as exc:
        raise ImportError(
            "RDKit is required for ECFP fingerprint generation. Run with "
            "`qsar/.venv_smiles/bin/python`."
        ) from exc
    return Chem, AllChem, DataStructs, rdFingerprintGenerator


def _load_torch_checkpoint(path: str | Path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


@dataclass(frozen=True)
class ECFPMoleculeFingerprint:
    smiles: str
    fingerprint: Tensor


@dataclass(frozen=True)
class BatchedECFPFingerprint:
    fingerprints: Tensor
    smiles: tuple[str, ...]

    @property
    def num_graphs(self) -> int:
        return len(self.smiles)

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> "BatchedECFPFingerprint":
        return BatchedECFPFingerprint(
            fingerprints=self.fingerprints.to(device, non_blocking=non_blocking),
            smiles=self.smiles,
        )


class ECFPFeaturizer:
    """Build fixed-length Morgan/ECFP fingerprints from SMILES."""

    def __init__(
        self,
        *,
        radius: int = 2,
        n_bits: int = 2048,
        use_chirality: bool = False,
    ) -> None:
        if radius < 0:
            raise ValueError("radius must be >= 0.")
        if n_bits <= 0:
            raise ValueError("n_bits must be > 0.")
        self.radius = int(radius)
        self.n_bits = int(n_bits)
        self.use_chirality = bool(use_chirality)

    def __call__(self, smiles: str) -> ECFPMoleculeFingerprint:
        Chem, AllChem, DataStructs, rdFingerprintGenerator = _load_rdkit()
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=self.use_chirality)
        bit_vector = self._morgan_fingerprint(mol, AllChem, rdFingerprintGenerator)
        values = torch.zeros((self.n_bits,), dtype=torch.float32)
        DataStructs.ConvertToNumpyArray(bit_vector, values.numpy())
        return ECFPMoleculeFingerprint(smiles=canonical_smiles, fingerprint=values)

    def _morgan_fingerprint(self, mol, AllChem, rdFingerprintGenerator):
        if hasattr(rdFingerprintGenerator, "GetMorganGenerator"):
            generator = rdFingerprintGenerator.GetMorganGenerator(
                radius=self.radius,
                fpSize=self.n_bits,
                includeChirality=self.use_chirality,
            )
            return generator.GetFingerprint(mol)
        return AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=self.radius,
            nBits=self.n_bits,
            useChirality=self.use_chirality,
        )


def collate_ecfp_fingerprints(items: Sequence[ECFPMoleculeFingerprint]) -> BatchedECFPFingerprint:
    if not items:
        raise ValueError("Cannot collate an empty ECFP batch.")
    return BatchedECFPFingerprint(
        fingerprints=torch.stack([item.fingerprint for item in items], dim=0),
        smiles=tuple(item.smiles for item in items),
    )


class ECFPFlashPointRegressor(nn.Module):
    """MLP baseline on top of fixed ECFP fingerprints."""

    def __init__(
        self,
        *,
        input_dim: int = 2048,
        hidden_dim: int = 512,
        num_layers: int = 3,
        readout_dim: int = 256,
        dropout: float = 0.1,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0.")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")

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

        self.encoder = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(current_dim, readout_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(readout_dim, max(readout_dim // 2, output_dim)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(readout_dim // 2, output_dim), output_dim),
        )

    def forward(self, batch: BatchedECFPFingerprint | Tensor) -> Tensor:
        fingerprints = batch.fingerprints if isinstance(batch, BatchedECFPFingerprint) else batch
        return self.head(self.encoder(fingerprints))


def build_ecfp_flash_point_model(**kwargs) -> ECFPFlashPointRegressor:
    return ECFPFlashPointRegressor(**kwargs)


class ECFPFlashPointPredictor:
    """Convenience wrapper for checkpointed ECFP flash-point models."""

    def __init__(
        self,
        model: ECFPFlashPointRegressor,
        *,
        featurizer: ECFPFeaturizer | None = None,
        target_columns: Sequence[str] = (),
        target_mean: Tensor | float | Sequence[float] | None = None,
        target_std: Tensor | float | Sequence[float] | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.featurizer = featurizer or ECFPFeaturizer()
        self.target_columns = tuple(target_columns)
        self.target_mean = self._as_tensor(target_mean)
        self.target_std = self._as_tensor(target_std)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
    ) -> "ECFPFlashPointPredictor":
        checkpoint = _load_torch_checkpoint(checkpoint_path, map_location=device or "cpu")
        model = build_ecfp_flash_point_model(**checkpoint["model_config"])
        predictor = cls(
            model,
            featurizer=ECFPFeaturizer(**checkpoint.get("featurizer_config", {})),
            target_columns=checkpoint.get("target_columns", checkpoint.get("property_names", ())),
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
        smiles: Iterable[str],
        *,
        batch_size: int = 64,
    ) -> list[dict[str, str | float]]:
        smiles_items = list(smiles)
        rows: list[dict[str, str | float] | None] = []

        for start in range(0, len(smiles_items), batch_size):
            batch_smiles = smiles_items[start : start + batch_size]
            fingerprints = []
            valid_smiles = []
            valid_positions = []
            for offset, item in enumerate(batch_smiles):
                position = start + offset
                try:
                    fingerprints.append(self.featurizer(item))
                    valid_smiles.append(item)
                    valid_positions.append(position)
                    rows.append(None)
                except ValueError:
                    rows.append({"smiles": item, "error": "invalid_smiles"})

            if not fingerprints:
                continue

            batch = collate_ecfp_fingerprints(fingerprints).to(self.device)
            prediction = self._inverse_transform(self.model(batch)).cpu()
            for item, values in zip(valid_smiles, prediction.tolist()):
                value_list = values if isinstance(values, list) else [values]
                row: dict[str, str | float] = {"smiles": item}
                row.update({name: float(value) for name, value in zip(self.target_columns, value_list)})
                rows[valid_positions.pop(0)] = row

        return [row for row in rows if row is not None]

    def _inverse_transform(self, prediction: Tensor) -> Tensor:
        if self.target_mean is None or self.target_std is None:
            return prediction
        return prediction * self.target_std.to(prediction.device) + self.target_mean.to(prediction.device)

    def _as_tensor(self, value: Tensor | float | Sequence[float] | None) -> Tensor | None:
        if value is None:
            return None
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

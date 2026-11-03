# Ester Oil Molecular Design

Reproducible workflows for ester-oil property prediction and molecular design using an external dataset supplied by the user.

This repository contains source code, experiment configurations, environment specifications, and reproducibility documentation. Datasets, pretrained weights, generated molecules, and experiment results are stored separately. The companion dataset is prepared as an independent Zenodo upload package; it has not yet been published and has no DOI.

The workflows cover **13 QSAR model configurations, batch prediction, autoencoder and diffusion-model training, multi-objective molecular search, and result processing**.

## Installation

Use Python 3.11. The pinned dependencies match the CPU environment used for validation. See [Reproducibility](docs/REPRODUCIBILITY.md) for GPU installation notes and reproducibility limits.

```bash
git clone https://github.com/chenshuohang/ester-oil-design.git
cd ester-oil-design
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r environments/requirements-qsar.txt
python -m pip install -e . --no-deps
```

For molecular generation, also install:

```bash
python -m pip install -r environments/requirements-generation.txt
```

## External data

The companion package, `ester-oil-data-v1.0.0.tar.gz`, is maintained separately from this repository. It contains the original property tables, permitted PubChem and BRICS500 structure collections, and 13 original QSAR baseline checkpoints. Until the Zenodo record is published, obtain this package directly from the project owner. No dataset or model-weight downloads are hosted as GitHub release assets.

Package SHA256: `db7432e0548811a17e2dc0497d2b72a501db1031dd65ee5b80205dc999897541`  
Package size: 50,935,696 bytes.

With the archive, `DATASET_README.md`, and `SHA256SUMS` in the same directory:

```bash
# Run this check from the directory containing the three package files.
cd /path/to/package-files
sha256sum -c SHA256SUMS
mkdir -p /path/to/datasets
tar -xzf ester-oil-data-v1.0.0.tar.gz -C /path/to/datasets
export ESTER_DATA=/path/to/datasets/ester-oil-data-v1.0.0
python -m ester_oil verify-data --data-root "$ESTER_DATA"
```

Return to the cloned repository before running the training commands below. The extracted data root contains `properties/`, `generation/`, `pretrained/`, and `manifest.json`. Verify the package before loading its checkpoints. See [Data](docs/DATA.md) for input columns, units, and package contents.

The newer ester dataset and its associated weights, predictions, and metrics are excluded. The external package excludes all contents of `data_qsar/` and `data_processing/fragmentation/from_160/`, as well as historical mixed training collections whose provenance could not be separated reliably.

## Property model training

Run a small training-and-prediction check for all 13 configurations:

```bash
python scripts/smoke_qsar.py --data-root "$ESTER_DATA" --output runs/qsar-smoke
```

Train individual models on the full task dataset:

```bash
python -m ester_oil train --task flash_point --model communicative_mpnn \
  --data-root "$ESTER_DATA" --output runs/flash-cmpnn --device cpu

python -m ester_oil train --task dynamic_viscosity --model conditional_communicative_mpnn \
  --data-root "$ESTER_DATA" --output runs/viscosity-cmpnn --device cpu
```

| Task | Available models | Prediction units |
|---|---|---|
| `flash_point` | `ecfp`, `directed_mpnn`, `communicative_mpnn`, `schnet` | K |
| `dielectric_constant` | `ecfp`, `directed_mpnn`, `communicative_mpnn`, `schnet` | Dimensionless |
| `dynamic_viscosity` | The four models above, plus `conditional_communicative_mpnn` | Pa·s; training uses the natural logarithm |

Configurations are in `configs/qsar/`. Defaults are seed `2026` and `300` epochs. Use `--epochs` to change the training duration. `--smoke` uses smaller networks and datasets for a one-epoch execution check. Supply a different dataset with `--input your.csv`, following the documented schema. Each run requires a new output directory.

Training writes `model.pt`, `model.metrics.json`, `split.json`, `config.json`, `train.log`, and `run_manifest.json`. Split indices refer to the dataset after invalid structures and labels have been filtered. Dynamic-viscosity splits are grouped by molecule, with molecule IDs recorded and overlap checked.

## Inference

Use a checkpoint from your own training run or an original baseline checkpoint from the external package:

```bash
python -m ester_oil predict --task flash_point \
  --checkpoint "$ESTER_DATA/pretrained/qsar/flash_point/communicative_mpnn.pt" \
  --smiles 'CCCCCCCC(=O)OCC' --output runs/prediction.csv

python -m ester_oil predict --task dynamic_viscosity \
  --checkpoint "$ESTER_DATA/pretrained/qsar/dynamic_viscosity/conditional_communicative_mpnn.pt" \
  --smiles 'CCCCCCCC(=O)OCC' --temperature 298.15 --pressure 101.325 \
  --output runs/viscosity-prediction.csv
```

For batch prediction, use `--input input.csv --smiles-column canonical_smiles`. Dynamic-viscosity predictions include temperature in K and pressure in kPa. To express flash point in degrees Celsius, subtract 273.15 from the prediction in K.

## Molecular generation

Train the generative models from the permitted external structure collections:

```bash
# Small CPU check: preparation, autoencoder training, encoding, and diffusion training.
JAX_PLATFORMS=cpu python -m ester_oil.generation all \
  --data-root "$ESTER_DATA" --work runs/generation-smoke --smoke

# Full configuration: 10 autoencoder epochs and 2,000,000 diffusion steps.
python -m ester_oil.generation all --data-root "$ESTER_DATA" \
  --work runs/generation --batch-size 16 --epochs 10 --steps 2000000
```

The stages are also available as `prepare`, `train-ae`, `encode`, and `train-dit`. Supply custom structure tables with `--input first.csv second.csv`; use `--limit` only for debugging. The full workflow combines the permitted PubChem and BRICS500 filtered structures, deduplicates standardized structure strings, and applies a fixed random split. Only the training partition is encoded for diffusion training; the held-out partition is excluded.

Historical mixed-corpus indices, latent arrays, and generative checkpoints are not distributed. Retraining from the permitted structures does not reproduce the original mixed-corpus training run, its model parameters, or its historical 30-generation candidate set.

## Multi-objective molecular search

After training the generative models:

```bash
python scripts/make_design_config.py --generation-run runs/generation \
  --data-root "$ESTER_DATA" --output runs/design-config.json
python -m ester_oil design --config runs/design-config.json --output runs/design
python -m cases.ester_oil_multi_prop.postprocess_results runs/design
```

The objectives are to increase flash point while reducing dynamic viscosity and dielectric constant. The search template is `configs/design.json`. Selection uses a fixed constrained NSGA-II implementation, independent of whether `pymoo` is installed. Meaningful candidates require adequately trained generative models; passing a small execution check does not establish candidate quality.

## Repository layout

| Path | Purpose |
|---|---|
| `ester_oil/` | Command-line interfaces, run manifests, data preparation, and selection logic |
| `qsar/` | Property model definitions and training implementations |
| `molsculptor/` | Molecular autoencoder, diffusion networks, and sampling code |
| `cases/ester_oil_multi_prop/` | Multi-objective search and result processing |
| `configs/` | Experiment settings, network architectures, and token alphabet |
| `environments/` | Pinned dependencies and tested environment details |
| `scripts/` | Model smoke checks and search-configuration helpers |
| `tests/` | Configuration and selection regression checks |
| `docs/` | Data contracts, reproduction instructions, and validation records |

## Validation

Completed checks include all 13 small QSAR training-and-prediction runs, inference with all 13 original baseline checkpoints, reconstruction of the three task tables, repeated CPU training with a fixed seed, and a small autoencoder-to-diffusion-to-search workflow. Run the lightweight regression checks with:

```bash
python -m unittest discover -s tests -v
```

Full-scale training and GPU reproducibility have not been rerun. The small generation check used a seed fallback and did not produce valid new candidates. See [Validation](docs/VALIDATION.md) for the exact scope and results. The detailed supporting documents are currently in Chinese.

## Provenance and licensing

See [Third-party notices](THIRD_PARTY_NOTICES.md) for code provenance and existing third-party licenses. A project-wide license has not yet been selected. The independent Zenodo record still requires author, affiliation, and dataset-license information before publication; no DOI has been assigned.

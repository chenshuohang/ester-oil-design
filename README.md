# Ester Oil Molecular Design

可从给定数据集运行的酯类油性质预测与分子生成代码仓。代码、数据和模型权重分开保存。本版本不包含新酯类数据集及其模型结果，也不包含 `data_processing/fragmentation/from_160/` 的文件或来源不清的历史混合训练数据。

本仓库提供 **13 个 QSAR 训练配置、统一预测入口、自编码器→扩散模型训练流程、多目标分子搜索与结果整理**。配套 Zenodo 包包含原有数据和 13 个原有 QSAR 基线权重。完整规模训练成本较高；小规模验证用于检查程序能否运行，不代表模型已经收敛。

## 1. 环境

使用 Python 3.11。以下 CPU 环境是实际检查使用的软件版本；GPU 安装说明见 [复现说明](docs/REPRODUCIBILITY.md)。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r environments/requirements-qsar.txt
python -m pip install -e . --no-deps
```

若还需要分子生成，再安装：

```bash
python -m pip install -r environments/requirements-generation.txt
```

## 2. 配套数据

配套数据包 `ester-oil-data-v1.0.0.tar.gz`、校验文件和数据说明从 [GitHub v1.0.0 发布附件](https://github.com/chenshuohang/ester-oil-design/releases/tag/v1.0.0) 下载。数据与代码分开存放，解压后把数据根目录传给 `--data-root`。该数据包也已整理为 Zenodo 上传包，目前尚未发布 Zenodo DOI。

数据包 SHA256：`db7432e0548811a17e2dc0497d2b72a501db1031dd65ee5b80205dc999897541`（50,935,696 字节）。

```bash
sha256sum -c /path/to/SHA256SUMS  # 在下载附件所在目录执行
mkdir -p /path/to/datasets
tar -xzf /path/to/ester-oil-data-v1.0.0.tar.gz -C /path/to/datasets
export ESTER_DATA=/path/to/datasets/ester-oil-data-v1.0.0
python -m ester_oil verify-data --data-root "$ESTER_DATA"
```

数据根目录有 `properties/`、`generation/`、`pretrained/` 和 `manifest.json`。所有二进制权重来自同一核验过的数据包，先校验再加载。详见 [数据说明](docs/DATA.md)。

## 3. 性质模型训练

先检查 13 套小规模训练与预测流程：

```bash
python scripts/smoke_qsar.py --data-root "$ESTER_DATA" --output runs/qsar-smoke
```

用完整数据训练某个模型：

```bash
python -m ester_oil train --task flash_point --model communicative_mpnn \
  --data-root "$ESTER_DATA" --output runs/flash-cmpnn --device cpu

python -m ester_oil train --task dynamic_viscosity --model conditional_communicative_mpnn \
  --data-root "$ESTER_DATA" --output runs/viscosity-cmpnn --device cpu
```

| 任务 | 可选模型 | 预测单位 |
|---|---|---|
| `flash_point` | `ecfp`, `directed_mpnn`, `communicative_mpnn`, `schnet` | K |
| `dielectric_constant` | 同上 | 无量纲 |
| `dynamic_viscosity` | 同上，加 `conditional_communicative_mpnn` | Pa·s，训练目标为自然对数 |

所有配置见 `configs/qsar/`，默认 seed=2026，默认训练 300 epochs。`--epochs` 可以覆盖训练轮数；`--smoke` 使用缩小的网络和数据，只训练 1 epoch。`--input your.csv` 可替换数据，字段要求见数据说明。输出目录必须是新目录，避免覆盖旧结果。

每次训练保存 `model.pt`、`model.metrics.json`、`split.json`、`config.json`、`train.log`、`run_manifest.json`。训练/验证划分记录的是过滤无效标签与结构后的索引；动力黏度按分子分组，记录两侧分子 ID 并检查无交叉。

## 4. 预测

可以加载自己训练的模型，也可以加载数据包中原有的基线模型：

```bash
python -m ester_oil predict --task flash_point \
  --checkpoint "$ESTER_DATA/pretrained/qsar/flash_point/communicative_mpnn.pt" \
  --smiles 'CCCCCCCC(=O)OCC' --output runs/prediction.csv

python -m ester_oil predict --task dynamic_viscosity \
  --checkpoint "$ESTER_DATA/pretrained/qsar/dynamic_viscosity/conditional_communicative_mpnn.pt" \
  --smiles 'CCCCCCCC(=O)OCC' --temperature 298.15 --pressure 101.325 \
  --output runs/viscosity-prediction.csv
```

CSV 批量预测使用 `--input input.csv --smiles-column canonical_smiles`。动力黏度预测同时写出温度 K 和压力 kPa；闪点换成摄氏度需减 273.15。

## 5. 从允许数据重新训练分子生成模型

```bash
# CPU 小模型：结构准备、自编码器训练、编码、扩散模型训练
JAX_PLATFORMS=cpu python -m ester_oil.generation all \
  --data-root "$ESTER_DATA" --work runs/generation-smoke --smoke

# 正式训练：默认 10 个 AE epochs、2,000,000 个 DiT steps
python -m ester_oil.generation all --data-root "$ESTER_DATA" \
  --work runs/generation --batch-size 16 --epochs 10 --steps 2000000
```

也可分别运行 `prepare`、`train-ae`、`encode`、`train-dit`。自定义输入使用 `--input first.csv second.csv`；`--limit` 仅适合调试。正式流程默认合并允许的 PubChem 与 BRICS500 筛选结构，经标准化字符串去重和固定随机划分后，只编码训练部分。保留的测试部分不加入扩散训练。

**这条重新训练流程不等于原历史混合语料训练。** 历史绝对路径索引、混合语料、潜变量和原有生成模型权重未随包发布。因此不会声称从本包能精确恢复历史 30 代候选或旧生成模型参数。

## 6. 多目标搜索与结果整理

```bash
python scripts/make_design_config.py --generation-run runs/generation \
  --data-root "$ESTER_DATA" --output runs/design-config.json
python -m ester_oil design --config runs/design-config.json --output runs/design
python -m cases.ester_oil_multi_prop.postprocess_results runs/design
```

目标为提高闪点、降低动力黏度和介电常数。搜索模板见 `configs/design.json`。本版本固定使用带约束的 NSGA-II 选择，避免是否安装 `pymoo` 导致算法自动切换。生成模型必须经过充分训练；仅通过小模型流程检查不足以得到有意义的候选分子。

## 7. 目录与验证

- `ester_oil/`：统一入口、训练审计、数据准备及固定选择算法。
- `qsar/`：原有性质模型定义和训练实现。
- `molsculptor/`：分子自编码器、扩散网络与采样代码。
- `configs/`：可移植实验配置、网络结构和词表。
- `scripts/`：13 模型流程检查和搜索配置生成工具。
- `tests/`：配置与选择算法的回归检查。
- `docs/VALIDATION.md`：本次实际验证范围与结果。

运行 `python -m unittest discover -s tests -v` 执行轻量检查。

来源与许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。本地导出未代作者选择项目整体许可证，Zenodo 作者、单位、许可证和 DOI 仍需在发布前填写。

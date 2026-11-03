# 配套数据与排除范围

## 允许的数据

| 内容 | 包内位置 | 实际规模 |
|---|---|---:|
| 原有单分子主表 | `properties/processed/compound_master.csv` | 610 个分子 |
| 原有性质观测长表 | `properties/processed/property_observations.csv` | 17,761 条，504 个有观测的分子 |
| 闪点训练主表 | `properties/processed/oil_like_multitask.csv` | 610 行，189 个闪点标签 |
| 闪点观测表 | `properties/tasks/flash_point/` | 210 条观测、189 个分子 |
| 动力黏度训练表 | `properties/tasks/dynamic_viscosity/` | 10,549 条、223 个分子 |
| 介电常数训练表 | `properties/tasks/dielectric_constant/` | 87 条、87 个分子 |
| PubChem 筛选结构 | `generation/pubchem_filtered_smiles.csv` | 295,242 行 |
| BRICS500 输入结构 | `generation/brics500_seed_structures.csv` | 实际 507 行，“500”是历史文件名 |
| BRICS500 筛选结构 | `generation/brics500_filtered_smiles.csv` | 72,576 行 |
| BRICS500 生成结构 | `generation/brics500_generated_smiles.smi` | 原始允许分支的 SMILES 行文件 |
| 原有 QSAR 模型 | `pretrained/qsar/` | 13 份权重及对应指标 |

模型文件只包含原有训练任务的参数及元数据；闪点模型的划分为 151/38，而不是新酯类数据集的 190/47。模型来源见 `pretrained/catalog.json`。目录中保存的原指标仅作为历史参考，本次没有重新运行 300 epochs 来重复这些数值。

## CSV 输入约定

- 闪点：`canonical_smiles`, `flash_point_K`。本版本标量模型过滤无标签和无效结构后按固定种子随机划分。
- 介电常数：`canonical_smiles`, `property_value_si`（无量纲）。
- 动力黏度：`canonical_smiles`, `compound_id`, `temperature_K`, `pressure_kPa`, `log_dynamic_viscosity`；最后一列为 `ln(黏度 / Pa·s)`。缺失压力沿用模型的 101.325 kPa 默认值。
- 分子生成：`canonical_smiles`、`smiles` 或 `SMILES` 中任一列。输入必须是单分子结构；正式生成数据采用 C/H/O 分子。

温度、压力条件不应混淆。动力黏度 Pa·s 与运动黏度 cSt 不是同一目标。`table_schema.json` 保存每张表的实际列名与行数，`properties/processed/source_registry.csv` 保存来源、访问与许可说明，观测表保留逐条来源字段。

## 严格排除

- `data_qsar/` 全部内容，包括新酯类表、清洗表、配置、权重、指标、预测结果和图片。
- `data_processing/fragmentation/from_160/` 全部内容。
- 来源不能明确拆分的 `id2sanitized_smi.csv`、全部历史训练索引、混合索引和历史潜变量。
- 历史 AE/DiT 权重、酯基数量微调结果和全部 `outputs/` 运行产物。
- 各种旧优化实验、原始缓存、日志、笔记本输出、分子动力学轨迹及引擎构建产物。
- 原始出版商文档和联网缓存；保留允许的结构/性质数据表及其来源引用。

排除按文件来源执行。一个分子如果独立出现在允许的 PubChem 或原有性质数据中，不因为其 SMILES 也在被排除数据中出现就删除；否则会改变独立允许来源的内容。所有复制文件都来自显式允许列表，并对被排除文件进行 SHA-256 重复检查。

## 数据使用边界

该数据库含直接实验、公共注释和相关式生成的数据；`data_origin`、`source_name`、`notes` 必须随数据保留。这里没有把所有记录都重新标成实验实测。部分原有 README 与当前表的数量不一致，发布说明以实际表格统计为准。

数据包尚无作者确认的统一许可证。已有第三方来源的授权说明保存在 source_registry 中，不因本次打包被替换。

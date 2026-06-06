# 第六届计图人工智能热身赛：CoraGCN

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Jittor](https://img.shields.io/badge/Jittor-GCN-orange)
![Task](https://img.shields.io/badge/Task-Node%20Classification-green)

> 使用 Jittor 与 JittorGeometric 在 Cora 引文网络上训练两层 GCN，完成节点分类并生成热身赛提交文件。当前本地复现的最佳验证集准确率记录为 `0.8160`。

## 目录

- [第六届计图人工智能热身赛：CoraGCN](#第六届计图人工智能热身赛coragcn)
  - [目录](#目录)
  - [项目结构](#项目结构)
  - [环境配置](#环境配置)
  - [系统配置](#系统配置)
  - [模型结构](#模型结构)
  - [训练配置](#训练配置)
  - [输出文件](#输出文件)
  - [训练](#训练)
  - [测试](#测试)
  - [实验结果](#实验结果)

## 项目结构

```text
.
├── data/                     # 数据文件
│   └── cora.pkl
├── src/                      # 核心训练、推理和导出逻辑
│   └── gcn.py
├── scripts/                  # 辅助脚本
│   ├── package_submission.py
│   └── validate_submission.py
├── models/                   # 训练过程中保存的模型 checkpoint
├── outputs/                  # 结果、日志和图片输出
│   ├── images/
│   └── results/
│       ├── result.json
│       └── result.zip
├── configs/
│   └── default.yaml          # 默认运行参数
├── requirements.txt
└── README.md
```

项目不再使用 `release/` 文件夹。数据统一放在 `data/`，核心函数放在 `src/`，辅助命令放在 `scripts/`，模型输出放在 `models/`，预测结果、压缩包、日志和图片放在 `outputs/`。

## 环境配置

本项目使用 Python 3.7，主要依赖包括 numpy、jittor、jittor-geometric 和 PyYAML。

```bash
conda create -n CoraGCN python=3.7 -y
pip install -r requirements.txt
```

## 系统配置

默认配置文件是 `configs/default.yaml`。

关键路径：

```yaml
data_path: data/cora.pkl
result_path: outputs/results/result.json
zip_path: outputs/results/result.zip
output_dir: outputs/latest
model_dir: models
```

CPU 环境默认参数：

```yaml
use_cuda: 0
use_jittor_geometric: 0
```

如果本机 CUDA 和 JittorGeometric 算子可用，可以把两项改为 `1`，或在命令行覆盖。

## 模型结构

模型实现位于 `src/gcn.py` 的 `GCNNet`：

- 第一层：`GCNConv(num_features, hidden_dim)`
- 激活函数：`ReLU`
- 正则化：`Dropout`
- 第二层：`GCNConv(hidden_dim, num_classes)`
- 输出：`log_softmax`

输入特征会按节点进行行归一化，图结构会经过 `gcn_norm` 加自环并归一化。若 JittorGeometric 不可用，脚本会自动切换到纯 Jittor 的 dense fallback。

## 训练配置

当前默认训练参数写在 `configs/default.yaml`：

```yaml
seed: 42
seeds: [13]
epochs: 260
hidden_dim: 256
dropout: 0.85
lr: 0.01
weight_decay: 0.0005
log_interval: 20
export_strategy: auto
```

说明：

- `seeds` 支持单个或多个随机种子；多个 seed 时会比较最佳单模型和 ensemble。
- `export_strategy` 可选 `auto`、`best`、`ensemble`。
- `save_checkpoints: 1` 时，最佳验证集 epoch 的模型会保存到 `models/gcn_seed_<seed>.pkl`。
- 命令行参数会覆盖 `configs/default.yaml` 中的同名配置。

## 输出文件

训练后默认生成：

```text
models/gcn_seed_<seed>.pkl
outputs/latest/config.json
outputs/latest/command.txt
outputs/latest/train.log
outputs/results/result.json
outputs/results/result.zip
```

`result.json` 只包含 `test_mask` 对应节点：

```json
{
  "1708": 0,
  "1709": 1,
  "1710": 3
}
```

`result.zip` 的根目录包含比赛提交要求的两个文件：

```text
gcn.py
result.json
```

## 训练

使用默认参数训练：

```bash
python3 src/gcn.py
```

指定配置文件：

```bash
python3 src/gcn.py --config configs/default.yaml
```

临时覆盖参数：

```bash
python3 src/gcn.py --epochs 300 --dropout 0.8 --seeds 13,42,2024
```

训练结束会自动导出 `outputs/results/result.json`，并打包生成 `outputs/results/result.zip`。

## 测试

校验提交结果格式：

```bash
python3 scripts/validate_submission.py
```

重新打包提交文件：

```bash
python3 scripts/package_submission.py
```

检查压缩包根目录：

```bash
python3 -c "import zipfile; print(zipfile.ZipFile('outputs/results/result.zip').namelist())"
```

期望输出为：

```text
['gcn.py', 'result.json']
```

## 实验结果

当前本地记录的最佳验证集准确率为 `0.8160`。默认配置使用 seed `13`、`260` 个 epoch、隐藏层维度 `256`、dropout `0.85`，并在 CPU fallback 下复现稳定训练流程。

实验日志会保存在 `outputs/latest/train.log`，每次运行的实际配置会保存到 `outputs/latest/config.json`，便于复现实验结果。

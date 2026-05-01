# Cora GCN with JittorGeometric

使用 Jittor 与 JittorGeometric 在 Cora 引文网络上训练两层 GCN，完成节点分类并生成热身赛提交文件。当前本地复现的最佳验证集准确率记录为 `0.8160`。

## Highlights

- 可复现训练：配置文件、随机种子、运行命令和训练日志都会落盘。
- 提交友好：一键生成并校验 `release/result.json` 与 `release/result.zip`。
- 分数优化：保留验证集最优 epoch，当前默认配置验证集准确率为 `0.8160`。
- 结构清爽：根目录保留项目入口，过程资料归档到 `docs/`，本地缓存和环境文件不进入版本控制。
- CPU 兜底：无 CUDA 环境时可通过 `JITTOR_USE_CUDA=0` 运行。

## Project Layout

```text
.
├── configs/                  # Training configuration
│   └── default.json
├── docs/                     # Challenge notes and project records
├── release/                  # Competition release workspace
│   ├── data/cora.pkl
│   ├── gcn.py
│   ├── result.json
│   └── result.zip
├── scripts/                  # Local automation scripts
│   ├── package_submission.py
│   └── validate_submission.py
├── Makefile
├── requirements.txt
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

## Installation

推荐使用 Python 3.8+。Jittor / JittorGeometric 的安装与 CUDA 版本强相关，若已有比赛环境可直接复用。

```bash
python3 -m pip install -r requirements.txt
```

CPU 环境建议运行时设置：

```bash
export JITTOR_USE_CUDA=0
```

## Data

比赛数据已放在：

```text
release/data/cora.pkl
```

数据字段包括 `x`、`y`、`edge_index`、`train_mask`、`val_mask`、`test_mask`、`num_classes`、`num_features`。默认配置使用相对路径 `data/cora.pkl`，因此训练命令需要在 `release/` 目录下执行，或用 `--data-path` 指定其他路径。

## Train

推荐从仓库根目录运行，默认使用当前验证集最优配置：

```bash
make train
```

等价的完整命令：

```bash
cd release
JITTOR_USE_CUDA=0 python3 gcn.py --config ../configs/default.json
```

常用参数可通过命令行覆盖配置文件，例如：

```bash
cd release
python3 gcn.py --config ../configs/default.json --seeds 42,7,13,21,100 --epochs 300 --hidden-dim 256 --dropout 0.85 --lr 0.01
```

每次运行会在 `release/outputs/latest/` 保存：

- `config.json`：实际配置
- `command.txt`：运行命令
- `train.log`：训练日志

## Validate and Package

训练结束后脚本会自动对 `test_mask` 节点推理并导出：

```text
release/result.json
release/result.zip
```

从仓库根目录校验提交格式：

```bash
make validate
```

只重新打包，不重新训练：

```bash
make package
```

检查压缩包根目录：

```bash
python3 -c "import zipfile; print(zipfile.ZipFile('release/result.zip').namelist())"
```

期望输出：

```text
['gcn.py', 'result.json']
```

## Result

指标为节点分类 Accuracy，即预测正确节点数除以评测节点数。训练阶段只使用 `train_mask`，调参观察 `val_mask`，导出的 `result.json` 只包含 `test_mask` 节点。

测试集真实标签在比赛数据中隐藏，本地无法直接计算测试集 Accuracy。项目以验证集最佳准确率、输出格式和提交包结构作为提交前检查依据。当前正式结果来自 seed `13`、dropout `0.85`，最佳验证集准确率为 `0.8160`。线上成绩可能因评测隐藏标签和运行环境存在差异。

## Submission Format

`result.json` 是 `{节点编号: 预测类别}` 字典：

```json
{
  "1708": 0,
  "1709": 1,
  "1710": 3
}
```

约束：

- key 为测试节点编号的字符串形式
- value 为 `0-6` 的整数类别
- 只包含 `test_mask` 对应节点
- `result.zip` 根目录只包含 `gcn.py` 和 `result.json`

## License

代码使用 MIT License。Cora 数据集与比赛材料请遵循赛题发布方要求。

# Cora GCN with JittorGeometric

使用 Jittor 与 JittorGeometric 在 Cora 引文网络上训练两层 GCN，完成节点分类并生成热身赛提交文件。当前本地复现的最佳验证集准确率记录为 `0.8020`。

## 项目结构

```text
.
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── requirements.txt
├── configs/
│   └── default.json
├── scripts/
│   └── validate_submission.py
└── release/
    ├── README.md
    ├── gcn.py
    ├── result.json
    ├── result.zip
    └── data/
        └── cora.pkl
```

## 环境安装

推荐使用 Python 3.8+。Jittor / JittorGeometric 的安装与 CUDA 版本强相关，若已有比赛环境可直接复用。

```bash
pip install -r requirements.txt
```

CPU 环境建议运行时设置：

```bash
export JITTOR_USE_CUDA=0
```

## 数据准备

比赛数据已放在：

```text
release/data/cora.pkl
```

数据字段包括 `x`、`y`、`edge_index`、`train_mask`、`val_mask`、`test_mask`、`num_classes`、`num_features`。默认配置使用相对路径 `data/cora.pkl`，因此训练命令需要在 `release/` 目录下执行，或用 `--data-path` 指定其他路径。

## 训练

使用默认配置训练 200 个 epoch，并生成 `result.json` 与 `result.zip`：

```bash
cd release
JITTOR_USE_CUDA=0 python gcn.py --config ../configs/default.json --seed 42
```

常用参数可通过命令行覆盖配置文件，例如：

```bash
python gcn.py --config ../configs/default.json --epochs 300 --hidden-dim 256 --dropout 0.8 --lr 0.01
```

每次运行会在 `release/outputs/latest/` 保存：

- `config.json`：实际配置
- `command.txt`：运行命令
- `train.log`：训练日志

## 评测 / 推理

训练脚本会在训练结束后自动对 `test_mask` 节点推理并导出：

```text
release/result.json
release/result.zip
```

本地提交格式校验：

```bash
cd ..
python scripts/validate_submission.py
```

检查压缩包根目录：

```bash
python -c "import zipfile; print(zipfile.ZipFile('release/result.zip').namelist())"
```

期望输出：

```text
['gcn.py', 'result.json']
```

## 结果说明

指标为节点分类 Accuracy，即预测正确节点数除以评测节点数。训练阶段只使用 `train_mask`，调参观察 `val_mask`，导出的 `result.json` 只包含 `test_mask` 节点。

测试集真实标签在比赛数据中隐藏，本地无法直接计算测试集 Accuracy。项目以验证集最佳准确率、输出格式和提交包结构作为提交前检查依据。线上成绩可能因评测隐藏标签和运行环境存在差异。

## 提交格式

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

## 许可证与声明

代码使用 MIT License。Cora 数据集与比赛材料请遵循赛题发布方要求。

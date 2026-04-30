# Cora GCN with JittorGeometric

使用 Jittor 与 JittorGeometric 在 Cora 引文网络上训练两层 GCN，完成节点分类并生成热身赛提交文件。项目包含完整训练脚本、数据说明、结果校验脚本和可直接提交的 `result.json` / `result.zip`。

## Highlights

- 两层 GCN：`1433 -> 256 -> 7`
- 标准 GCN 归一化：自环 + 对称度归一化
- 仅使用 `train_mask` 训练，使用 `val_mask` 观察模型表现
- 仅对 `test_mask` 节点导出预测
- 本地 CPU 环境可运行，GPU / JittorGeometric 环境优先使用官方算子
- 当前复现验证集最佳准确率：`0.8020`

## Project Layout

```text
.
├── README.md
├── WORKFLOW.md
├── demand.md
├── requirements.txt
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

## Quick Start

推荐在已有 Jittor 环境中运行。当前机器使用的环境示例：

```bash
conda activate jittor_env
cd release
JITTOR_USE_CUDA=0 python gcn.py
```

如果有可用 CUDA 与完整 JittorGeometric GPU 算子，可直接运行：

```bash
cd release
python gcn.py
```

脚本会训练 200 个 epoch，并在结束后生成：

- `release/result.json`
- `release/result.zip`

## Dependencies

核心依赖：

- Python
- NumPy
- Jittor
- JittorGeometric

详见 [requirements.txt](requirements.txt)。Jittor / JittorGeometric 的安装与 CUDA 版本强相关，建议优先参考官方安装说明。

## Dataset

数据文件位于 `release/data/cora.pkl`，包含：

| Field | Shape / Type | Description |
|---|---|---|
| `x` | `2708 x 1433` | 节点词袋特征 |
| `y` | `2708` | 节点标签，测试集标签为 `-1` |
| `edge_index` | `2 x E` | COO 边列表 |
| `train_mask` | `2708` | 训练集掩码 |
| `val_mask` | `2708` | 验证集掩码 |
| `test_mask` | `2708` | 测试集掩码 |
| `num_classes` | `int` | 类别数，预期为 7 |
| `num_features` | `int` | 特征维度，预期为 1433 |

## Reproduce

1. 进入发布目录：

   ```bash
   cd release
   ```

2. 训练并生成预测：

   ```bash
   JITTOR_USE_CUDA=0 python gcn.py
   ```

3. 校验提交文件：

   ```bash
   cd ..
   python scripts/validate_submission.py
   ```

4. 检查压缩包结构：

   ```bash
   python -c "import zipfile; print(zipfile.ZipFile('release/result.zip').namelist())"
   ```

   期望输出：

   ```text
   ['gcn.py', 'result.json']
   ```

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

## Notes

`release/gcn.py` 会优先使用 JittorGeometric 的 `GCNConv`、`gcn_norm` 与稀疏格式转换。在本机 CPU 环境中，如果已安装的 JittorGeometric 包导入 CUDA-only 算子失败，脚本会自动切换到纯 Jittor fallback，以相同 GCN 公式完成训练和预测。

## License

代码使用 MIT License。数据集与比赛材料请遵循赛题发布方要求。

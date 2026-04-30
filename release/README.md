# Release Package

该目录是热身赛提交包工作区，包含训练脚本、数据文件、预测结果和最终压缩包。

## Files

```text
release/
├── README.md
├── gcn.py
├── result.json
├── result.zip
└── data/
    └── cora.pkl
```

## Run

在 CPU 环境运行：

```bash
JITTOR_USE_CUDA=0 python gcn.py
```

在可用 CUDA / JittorGeometric GPU 算子环境运行：

```bash
python gcn.py
```

训练结束后会生成 `result.json`。当前本地复现最佳验证集准确率为 `0.8020`。

## Result Format

`result.json` 只包含 `test_mask` 对应节点：

```json
{
  "1708": 0,
  "1709": 1,
  "1710": 3
}
```

要求：

- key 是节点编号字符串
- value 是 `0-6` 的整数类别
- 不包含训练集或验证集节点

## Package

`result.zip` 的根目录必须是：

```text
gcn.py
result.json
```

可以用下面命令检查：

```bash
python -c "import zipfile; print(zipfile.ZipFile('result.zip').namelist())"
```

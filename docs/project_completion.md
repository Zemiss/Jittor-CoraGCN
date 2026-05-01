# Cora GCN 热身赛项目完结文档

## 1. 项目结论

本项目已按 `docs/challenge.md` 的要求完成 Cora 引文网络节点分类热身赛任务：

- 已读取比赛数据 `release/data/cora.pkl`；
- 已使用 Jittor / JittorGeometric 风格的两层 GCN 完成节点分类建模；
- 已实现训练、验证、测试集预测与结果保存流程；
- 已生成符合提交格式的 `release/result.json`；
- 已生成提交压缩包 `release/result.zip`；
- 已提供本地提交格式校验脚本 `scripts/validate_submission.py`；
- 当前记录的最佳验证集准确率为 `0.8160`，高于赛题要求的 `0.70` 通过线。

说明：测试集真实标签在比赛数据中被隐藏，本地无法直接计算测试集 Accuracy。项目以验证集表现、提交格式完整性和输出约束作为提交前检查依据。

## 2. 需求满足情况核对

| 需求项 | 实现位置 | 完成情况 |
|---|---|---|
| 读取 `data/cora.pkl` | `release/gcn.py` | 已完成 |
| 构建 GCN 模型 | `release/gcn.py` 的 `GCNNet` | 已完成 |
| 使用训练集训练 | `release/gcn.py` 的 `train()` | 已完成 |
| 使用验证集评估 | `release/gcn.py` 的 `evaluate()` | 已完成 |
| 对测试集节点预测 | `release/gcn.py` 结果导出部分 | 已完成 |
| 生成 `result.json` | `release/result.json` | 已完成 |
| JSON key 使用节点编号字符串 | `release/result.json` | 已完成 |
| JSON value 为 `0-6` 类别整数 | `release/result.json` | 已完成 |
| 只包含 `test_mask` 节点 | `scripts/validate_submission.py` 校验通过 | 已完成 |
| 打包 `gcn.py` 与 `result.json` | `release/result.zip` | 已完成 |

已执行的提交文件校验结果：

```text
result_count_matches_test_mask: PASS
result_keys_match_test_mask: PASS
values_are_int: PASS
values_are_in_0_6: PASS
zip_root_files: PASS
```

数据与结果规模：

- 节点数：`2708`
- 特征维度：`1433`
- 类别数：`7`
- 边索引规模：`2 x 10858`
- 训练 / 验证 / 测试节点数：`140 / 500 / 1000`
- `result.json` 预测数量：`1000`
- 预测类别集合：`[0, 1, 2, 3, 4, 5, 6]`

## 3. 如何入手项目

建议从以下顺序阅读和运行：

1. 阅读需求：`docs/challenge.md`
2. 阅读工作流：`docs/workflow.md`
3. 查看总览：`README.md`
4. 进入提交目录：`release/`
5. 阅读核心脚本：`release/gcn.py`
6. 运行训练并生成结果：`JITTOR_USE_CUDA=0 python3 gcn.py`
7. 回到项目根目录校验提交：`python3 scripts/validate_submission.py`
8. 检查提交包：确认 `release/result.zip` 根目录只有 `gcn.py` 和 `result.json`

如果本机配置了 CUDA 且 JittorGeometric GPU 算子可正常导入，也可以在 `release/` 下直接运行：

```bash
python3 gcn.py
```

如果本机没有可用 CUDA，推荐运行：

```bash
cd release
JITTOR_USE_CUDA=0 python3 gcn.py
```

## 4. 项目目录说明

```text
.
├── README.md
├── requirements.txt
├── scripts/
│   └── validate_submission.py
├── docs/
│   ├── demand.md
│   ├── WORKFLOW.md
│   └── PROJECT_COMPLETION.md
└── release/
    ├── README.md
    ├── gcn.py
    ├── result.json
    ├── result.zip
    └── data/
        └── cora.pkl
```

各目录职责：

- `docs/`：保存需求、执行流程和项目完结文档；
- `release/`：保存比赛提交相关文件，是训练脚本的运行目录；
- `scripts/`：保存本地辅助校验脚本；
- `README.md`：项目入口说明；
- `requirements.txt`：依赖声明。

## 5. 核心功能实现形式

### 5.1 数据加载

`release/gcn.py` 使用 pickle 读取：

```text
release/data/cora.pkl
```

默认配置使用相对路径：

```json
"data_path": "data/cora.pkl"
```

因此通常在 `release/` 目录运行训练；也可以通过 `--data-path` 指定其他路径。

读取后会把数据转换为 Jittor 张量，并放入轻量的 `GraphData` 对象中，字段包括：

- `x`：节点特征；
- `y`：节点标签；
- `edge_index`：图边；
- `train_mask`：训练集掩码；
- `val_mask`：验证集掩码；
- `test_mask`：测试集掩码。

### 5.2 特征处理

脚本对节点特征做行归一化：

```python
row_sum = data.x.sum(dim=1, keepdims=True)
row_sum = jt.clamp(row_sum, min_v=1e-12)
data.x = data.x / row_sum
```

这样每个节点的词袋特征会被标准化，符合 Cora 节点分类中常见的预处理方式。

### 5.3 图归一化

脚本调用 `gcn_norm` 对边进行 GCN 标准归一化：

- 添加自环；
- 计算度矩阵；
- 使用对称归一化；
- 生成 GCN 卷积所需的边权重。

在 JittorGeometric 可用时，会使用官方 `gcn_norm`、`cootocsc`、`cootocsr`。在 CPU 环境下如果 JittorGeometric 的 CUDA-only 算子导入失败，脚本会切换到纯 Jittor fallback，使用同样的 GCN 公式构造稠密邻接矩阵。

### 5.4 模型结构

模型定义在 `GCNNet` 中，是两层 GCN：

```text
1433 -> 256 -> 7
```

前向流程：

1. 第一层 GCN 卷积；
2. ReLU 激活；
3. Dropout；
4. 第二层 GCN 卷积；
5. `log_softmax` 输出每个节点的类别对数概率。

该结构简洁，符合赛题“使用 GCN 完成节点分类”的要求。

### 5.5 训练逻辑

训练函数 `train()` 只使用 `train_mask` 对应节点：

- 前向得到所有节点预测；
- 取训练节点预测；
- 取训练节点标签；
- 用 `nn.nll_loss` 计算损失；
- 用 Adam 优化器更新参数。

优化器配置：

```text
Adam(lr=0.01, weight_decay=5e-4)
```

脚本默认训练 `260` 个 epoch。当前配置使用 seed `13`、dropout `0.85`，并保留验证集最优 epoch 的预测概率。

### 5.6 验证逻辑

验证函数 `evaluate()` 会在训练集和验证集上计算准确率：

- 使用 `argmax` 得到预测类别；
- 分别按 `train_mask` 和 `val_mask` 统计正确数量；
- 返回训练准确率和验证准确率。

脚本记录 `best_val_acc`，用于观察模型调参效果。当前项目记录的最佳验证准确率为 `0.8160`。

### 5.7 测试集预测与导出

训练结束后，脚本会：

1. 对全部节点进行预测；
2. 使用 `raw['test_mask']` 找出测试集节点编号；
3. 构造 `{节点编号字符串: 预测类别整数}`；
4. 写入 `release/result.json`。

输出格式示例：

```json
{
  "1708": 0,
  "1709": 1,
  "1710": 3
}
```

### 5.8 提交校验

`scripts/validate_submission.py` 会检查：

- `result.json` 数量是否等于测试节点数量；
- key 是否完全等于 `test_mask` 节点编号；
- value 是否都是整数；
- value 是否全部在 `0-6` 范围内；
- `result.zip` 根目录是否只有 `gcn.py` 和 `result.json`。

推荐每次重新训练或修改 `result.json` 后都运行一次。

## 6. 运行与提交流程

完整复现流程如下：

```bash
cd release
JITTOR_USE_CUDA=0 python3 gcn.py
cd ..
python3 scripts/validate_submission.py
```

提交文件位于：

```text
release/result.zip
```

压缩包内部应为：

```text
gcn.py
result.json
```

不要把整个 `release/` 目录打进压缩包。

## 7. 后续维护建议

- 若更换环境，优先确认 Jittor 与 JittorGeometric 能正常导入；
- 若重新训练生成 `result.json`，需要同步重新生成 `result.zip`；
- 若修改模型参数，应记录新的验证集最佳准确率；
- 若提交前只做格式检查，运行 `scripts/validate_submission.py` 即可；
- 若追求更高准确率，可优先微调 `hidden_dim`、`dropout`、`lr`、`weight_decay` 和训练 epoch。

## 8. 最终产物清单

- `release/gcn.py`：训练、验证、预测和导出脚本；
- `release/result.json`：测试集预测结果；
- `release/result.zip`：比赛提交压缩包；
- `scripts/validate_submission.py`：提交格式校验脚本；
- `README.md`：项目总览和快速开始；
- `docs/workflow.md`：开发与复现工作流；
- `docs/project_completion.md`：项目完结说明。

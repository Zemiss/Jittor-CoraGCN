# Cora GCN 热身赛工作流

## 目标

基于 `release/data/cora.pkl`，使用 Jittor 与 JittorGeometric 训练 GCN 节点分类模型，对 `test_mask` 对应节点生成预测结果，最终提交 `gcn.py`、`result.json` 和打包文件 `result.zip`。

通过标准：

- `result.json` 格式正确；
- 仅包含测试集节点预测；
- 预测类别为 `0-6` 的整数；
- 测试集准确率达到或超过 `0.70`。

## 工作流总览

1. 准备运行环境
2. 检查数据与代码框架
3. 补全 `gcn.py` 中的 TODO
4. 训练 GCN 模型并观察验证集表现
5. 生成 `result.json`
6. 校验提交文件
7. 打包为 `result.zip`

## 1. 准备运行环境

需要安装：

- Python
- Jittor
- JittorGeometric
- NumPy

如果使用 GPU，确认 CUDA 环境可用，因为示例代码中设置了：

```python
jt.flags.use_cuda = 1
```

如果本机没有可用 GPU，可先将其改为：

```python
jt.flags.use_cuda = 0
```

验证目标：

- 可以正常导入 `jittor`；
- 可以正常导入 `jittor_geometric`；
- 运行脚本时不会因为 CUDA 或依赖缺失退出。

## 2. 检查数据与代码框架

当前发布包结构应为：

```text
release/
  gcn.py
  README.md
  data/
    cora.pkl
```

数据文件 `cora.pkl` 应包含：

- `x`：节点特征，形状为 `2708 x 1433`
- `y`：节点标签，测试集标签为 `-1`
- `edge_index`：图边列表
- `train_mask`：训练集节点掩码
- `val_mask`：验证集节点掩码
- `test_mask`：测试集节点掩码
- `num_classes`：类别数，预期为 `7`
- `num_features`：特征维度，预期为 `1433`

验证目标：

- `release/data/cora.pkl` 存在；
- `release/gcn.py` 能读取 `data/cora.pkl`；
- 脚本运行目录应为 `release/`，否则相对路径 `data/cora.pkl` 会找不到。

## 3. 补全模型与训练逻辑

在 `release/gcn.py` 中补全 TODO，建议按以下顺序处理。

### 3.1 定义 GCN 模型

在 `GCNNet.__init__` 中定义两层 GCN：

- 第一层：`num_features -> hidden_dim`
- 第二层：`hidden_dim -> num_classes`

验证目标：

- `model = GCNNet(...)` 可以正常初始化；
- `model.parameters()` 能被优化器读取。

### 3.2 实现前向传播

在 `GCNNet.execute` 中完成：

1. 第一层 GCN 卷积；
2. ReLU 激活；
3. Dropout；
4. 第二层 GCN 卷积。

验证目标：

- `model()` 输出形状为 `2708 x 7`；
- 输出每一行对应一个节点的 7 类 logits。

### 3.3 实现训练函数

训练函数应完成：

1. 前向传播；
2. 只取 `train_mask` 对应节点；
3. 使用真实标签计算交叉熵损失；
4. 调用优化器更新参数。

验证目标：

- 每个 epoch 可以完成一次参数更新；
- loss 计算只使用训练集节点，不使用验证集或测试集标签。

### 3.4 实现验证函数

验证函数应分别计算：

- 训练集准确率；
- 验证集准确率。

验证目标：

- 使用 `argmax` 得到预测类别；
- 准确率计算只在当前 mask 对应节点上进行；
- 不使用 `test_mask` 参与调参。

## 4. 训练模型

在 `release/` 目录下运行：

```bash
python gcn.py
```

脚本默认训练 `200` 个 epoch，并每 `20` 个 epoch 打印训练集准确率和最佳验证集准确率。

观察重点：

- 训练过程是否正常完成；
- 验证集准确率是否持续提升；
- 最终验证集准确率是否接近或超过可接受水平。

如验证集表现不足，可优先尝试小范围调整：

- `hidden_dim`
- `dropout`
- `lr`
- `weight_decay`
- `epoch` 数量

不建议一开始引入复杂模型或大规模重构，先确保基础 GCN 跑通。

## 5. 生成 `result.json`

训练完成后，脚本应：

1. 使用最终模型对全部节点预测；
2. 取 `test_mask` 对应节点编号；
3. 生成 `{节点编号: 预测类别}` 字典；
4. 保存为 `release/result.json`。

格式要求：

```json
{
  "1708": 0,
  "1709": 1,
  "1710": 3
}
```

注意：

- JSON 的 key 使用字符串形式；
- value 使用整数类别；
- 只输出测试集节点；
- 不要输出训练集或验证集节点。

## 6. 提交前检查

提交前逐项检查：

- [ ] 使用 Jittor 与 JittorGeometric 完成训练；
- [ ] `release/data/cora.pkl` 能被正确读取；
- [ ] `release/gcn.py` 中没有未完成的 TODO；
- [ ] `release/result.json` 已生成；
- [ ] `result.json` 的 key 是测试集节点编号字符串；
- [ ] `result.json` 的 value 是 `0-6` 之间的整数；
- [ ] `result.json` 只包含 `test_mask` 对应节点；
- [ ] 本地运行 `python gcn.py` 可以完整结束；
- [ ] 打包文件根目录包含 `gcn.py` 和 `result.json`。

## 7. 打包提交

进入 `release/` 目录后，将以下文件放入压缩包根目录：

```text
gcn.py
result.json
```

最终产物：

```text
result.zip
```

提交包内部结构应为：

```text
result.zip
  gcn.py
  result.json
```

不要打包成：

```text
result.zip
  release/
    gcn.py
    result.json
```

## 推荐执行顺序

```text
检查环境
  -> 检查 data/cora.pkl
  -> 补全 gcn.py 的模型定义
  -> 补全训练与验证逻辑
  -> 运行 python gcn.py
  -> 观察验证集准确率
  -> 生成 result.json
  -> 检查 JSON 格式和节点范围
  -> 打包 result.zip
```

'''
热身赛：基于 GCN 的 Cora 节点分类任务
参赛选手需要填充注释为 TODO 的部分完成该赛题。
'''

import os.path as osp
import os
import json
import pickle

# 本机无可用 CUDA 时，用 `JITTOR_USE_CUDA=0 python gcn.py` 可强制走 CPU。
if os.environ.get('JITTOR_USE_CUDA') == '0':
    os.environ.setdefault('nvcc_path', '')

import numpy as np
import jittor as jt
from jittor import nn

USING_JITTOR_GEOMETRIC = True

try:
    from jittor_geometric.nn import GCNConv
    from jittor_geometric.ops import cootocsr, cootocsc
    from jittor_geometric.nn.conv.gcn_conv import gcn_norm
except Exception as exc:
    if os.environ.get('JITTOR_USE_CUDA') != '0':
        raise

    USING_JITTOR_GEOMETRIC = False
    print(f'JittorGeometric CPU 导入失败，切换到纯 Jittor fallback: {exc}')

    def gcn_norm(edge_index, edge_weight=None, num_nodes=None,
                 improved=False, add_self_loops=True):
        edge_index_np = edge_index.numpy()
        if edge_weight is None:
            edge_weight_np = np.ones(edge_index_np.shape[1], dtype=np.float32)
        else:
            edge_weight_np = edge_weight.numpy().astype(np.float32)

        if add_self_loops:
            loop_index = np.arange(num_nodes, dtype=np.int64)
            loop_edges = np.vstack([loop_index, loop_index])
            loop_weight = np.full(num_nodes, 2.0 if improved else 1.0,
                                  dtype=np.float32)
            edge_index_np = np.concatenate([edge_index_np, loop_edges], axis=1)
            edge_weight_np = np.concatenate([edge_weight_np, loop_weight])

        row, col = edge_index_np
        deg = np.zeros(num_nodes, dtype=np.float32)
        np.add.at(deg, col, edge_weight_np)
        deg_inv_sqrt = np.zeros_like(deg)
        deg_inv_sqrt[deg > 0] = np.power(deg[deg > 0], -0.5)
        norm_weight = deg_inv_sqrt[row] * edge_weight_np * deg_inv_sqrt[col]
        return jt.array(edge_index_np.astype(np.int64)), jt.array(norm_weight)

    def cootocsc(edge_index, edge_weight, v_num):
        return None

    def cootocsr(edge_index, edge_weight, v_num):
        return None

    class GCNConv(nn.Module):
        def __init__(self, in_channels, out_channels, bias=True):
            super(GCNConv, self).__init__()
            self.weight = jt.randn((in_channels, out_channels))
            self.bias = jt.zeros((out_channels,)) if bias else None

        def execute(self, x, csc, csr):
            out = data.adj @ (x @ self.weight)
            if self.bias is not None:
                out = out + self.bias
            return out


# ============================================================
# 基本配置
# ============================================================
jt.flags.use_cuda = int(os.environ.get('JITTOR_USE_CUDA', '1'))
jt.misc.set_global_seed(42)

# ============================================================
# 第一步：加载数据集
# ============================================================
data_path = osp.join('data', 'cora.pkl')

with open(data_path, 'rb') as f:
    raw = pickle.load(f)

# 将 numpy 数据转为 jittor 张量，构造 data 对象
class GraphData:
    pass

data = GraphData()
data.x = jt.array(raw['x'].astype(np.float32))
data.y = jt.array(raw['y'].astype(np.int64))
data.edge_index = jt.array(raw['edge_index'].astype(np.int64))
data.train_mask = jt.array(raw['train_mask'])
data.val_mask = jt.array(raw['val_mask'])
data.test_mask = jt.array(raw['test_mask'])
num_features = raw['num_features']
num_classes = raw['num_classes']

# 对特征做行归一化（等同于 T.NormalizeFeatures()）
row_sum = data.x.sum(dim=1, keepdims=True)
row_sum = jt.clamp(row_sum, min_v=1e-12)
data.x = data.x / row_sum

# ============================================================
# 第二步：图的边归一化 + 稀疏格式转换
# ============================================================
v_num = data.x.shape[0]
edge_index, edge_weight = data.edge_index, None

edge_index, edge_weight = gcn_norm(
    edge_index, edge_weight, v_num,
    improved=False, add_self_loops=True
)

# 将 COO 格式转换为 CSC 和 CSR 稀疏矩阵格式
with jt.no_grad():
    data.csc = cootocsc(edge_index, edge_weight, v_num)
    data.csr = cootocsr(edge_index, edge_weight, v_num)

if not USING_JITTOR_GEOMETRIC:
    dense_adj = np.zeros((v_num, v_num), dtype=np.float32)
    edge_index_np = edge_index.numpy()
    edge_weight_np = edge_weight.numpy()
    dense_adj[edge_index_np[1], edge_index_np[0]] = edge_weight_np
    data.adj = jt.array(dense_adj)

# ============================================================
# 第三步：定义 GCN 模型
# ============================================================
class GCNNet(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=256, dropout=0.8):
        super(GCNNet, self).__init__()
        self.dropout = dropout

        # 定义两层 GCN 卷积层
        # 提示：GCNConv(in_channels, out_channels)
        self.conv1 = GCNConv(in_channels=num_features, out_channels=hidden_dim)
        self.conv2 = GCNConv(in_channels=hidden_dim, out_channels=num_classes)

    def execute(self):
        x, csc, csr = data.x, data.csc, data.csr

        # 实现前向传播
        # 1. 第一层 GCN 卷积 + ReLU 激活
        # 2. Dropout
        # 3. 第二层 GCN 卷积
        x = nn.relu(self.conv1(x, csc, csr))
        x = nn.dropout(x, self.dropout, is_train=self.training)
        x = self.conv2(x, csc, csr)

        return nn.log_softmax(x, dim=1)

# 初始化模型和优化器
model = GCNNet(
    num_features=num_features,
    num_classes=num_classes,
    hidden_dim=256,
    dropout=0.8
)
optimizer = nn.Adam(params=model.parameters(), lr=0.01, weight_decay=5e-4)

# ============================================================
# 第四步：定义训练函数
# ============================================================
def train():
    model.train()

    # 1. 前向传播
    pred = model()[data.train_mask]

    # 实现训练逻辑的以下 2-4 步骤
    # 2. 获取训练集节点的预测值
    # 3. 计算交叉熵损失
    # 4. 反向传播更新参数
    label = data.y[data.train_mask]
    loss = nn.nll_loss(pred, label)
    optimizer.step(loss)

# ============================================================
# 第五步：定义测试函数
# ============================================================
def test():
    model.eval()
    logits = model()
    accs = []

    for mask in [data.train_mask, data.val_mask]:
        # 实现预测和准确率计算
        # 1. 使用 jt.argmax 获取预测类别
        pred, _ = jt.argmax(logits[mask], dim=1)
        # 2. 计算预测准确率
        correct = (pred == data.y[mask]).sum().item()
        total = mask.sum().item()
        acc = correct / total
        accs.append(acc)

    return accs

# ============================================================
# 第六步：训练模型
# ============================================================
best_val_acc = 0

for epoch in range(1, 201):
    train()
    train_acc, val_acc = test()

    if val_acc > best_val_acc:
        best_val_acc = val_acc

    if epoch % 20 == 0:
        log = 'Epoch: {:03d}, Train Acc: {:.4f}, Val Acc: {:.4f}'
        print(log.format(epoch, train_acc, best_val_acc))

print(f'\n最终结果: Val Acc: {best_val_acc:.4f}')

# ============================================================
# 第七步：生成并保存预测结果
# ============================================================
model.eval()

# 1. 使用训练好的模型对所有节点进行预测
logits = model()
pred, _ = jt.argmax(logits, dim=1)
pred = pred.numpy()

# 2. 提取测试集节点的预测类别
test_indices = np.nonzero(raw['test_mask'])[0]

# 3. 构建字典 {节点编号: 预测类别}
result = {}
for idx in test_indices:
    result[str(int(idx))] = int(pred[int(idx)])

# 4. 保存为 result.json
output_path = 'result.json'
with open(output_path, 'w') as f:
    json.dump(result, f, indent=2)

print(f"预测结果已保存到 {output_path}")
print(f"共预测 {len(result)} 个测试节点")

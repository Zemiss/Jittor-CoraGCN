"""Cora GCN node classification with Jittor / JittorGeometric.

The script trains a two-layer GCN, reports train/validation accuracy,
exports predictions for test_mask nodes, and writes reproducibility metadata.
"""

import argparse
import json
import os
import pickle
import random
import sys
import zipfile
from pathlib import Path


def requested_cpu_from_argv():
    """Detect CPU mode before importing JittorGeometric CUDA-related modules."""
    if "--use-cuda" in sys.argv:
        index = sys.argv.index("--use-cuda")
        if index + 1 < len(sys.argv):
            return sys.argv[index + 1] == "0"

    if "--config" in sys.argv:
        index = sys.argv.index("--config")
        if index + 1 < len(sys.argv):
            config_path = Path(sys.argv[index + 1])
            if config_path.exists():
                with config_path.open(encoding="utf-8") as f:
                    return json.load(f).get("use_cuda") == 0
    return False


def requested_dense_fallback_from_argv():
    """Detect whether to skip JittorGeometric before importing it."""
    if os.environ.get("JITTOR_DENSE_FALLBACK") == "1":
        return True

    if "--use-jittor-geometric" in sys.argv:
        index = sys.argv.index("--use-jittor-geometric")
        if index + 1 < len(sys.argv):
            return sys.argv[index + 1] == "0"

    if "--config" in sys.argv:
        index = sys.argv.index("--config")
        if index + 1 < len(sys.argv):
            config_path = Path(sys.argv[index + 1])
            if config_path.exists():
                with config_path.open(encoding="utf-8") as f:
                    return json.load(f).get("use_jittor_geometric") == 0
    return False


# 本机无可用 CUDA 时，用 `JITTOR_USE_CUDA=0 python gcn.py` 或 `--use-cuda 0`
# 可强制走 CPU fallback。
if os.environ.get("JITTOR_USE_CUDA") == "0" or requested_cpu_from_argv():
    os.environ["JITTOR_USE_CUDA"] = "0"
    os.environ.setdefault("nvcc_path", "")

import numpy as np
import jittor as jt
from jittor import nn

USING_JITTOR_GEOMETRIC = not requested_dense_fallback_from_argv()

try:
    if not USING_JITTOR_GEOMETRIC:
        raise ImportError("disabled by --use-jittor-geometric 0")
    from jittor_geometric.nn import GCNConv
    from jittor_geometric.ops import cootocsr, cootocsc
    from jittor_geometric.nn.conv.gcn_conv import gcn_norm
except Exception as exc:
    USING_JITTOR_GEOMETRIC = False
    print(f"JittorGeometric 导入失败，切换到纯 Jittor fallback: {exc}")

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
            out = csc @ (x @ self.weight)
            if self.bias is not None:
                out = out + self.bias
            return out


DEFAULT_CONFIG = {
    "data_path": "data/cora.pkl",
    "result_path": "result.json",
    "zip_path": "result.zip",
    "output_dir": "outputs/latest",
    "seed": 42,
    "seeds": None,
    "epochs": 200,
    "hidden_dim": 256,
    "dropout": 0.8,
    "lr": 0.01,
    "weight_decay": 5e-4,
    "log_interval": 20,
    "use_cuda": int(os.environ.get("JITTOR_USE_CUDA", "1")),
    "use_jittor_geometric": 1,
    "export_strategy": "auto",
}


def parse_args():
    """Parse command line arguments; CLI values override config values."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to a JSON config file.")
    parser.add_argument("--data-path", default=None, help="Path to cora.pkl.")
    parser.add_argument("--result-path", default=None, help="Output result.json path.")
    parser.add_argument("--zip-path", default=None, help="Output result.zip path.")
    parser.add_argument("--output-dir", default=None, help="Directory for logs/metadata.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seeds for validation-selected ensemble, e.g. 42,7,13.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs.")
    parser.add_argument("--hidden-dim", type=int, default=None, help="GCN hidden size.")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout ratio.")
    parser.add_argument("--lr", type=float, default=None, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Adam weight decay.")
    parser.add_argument("--log-interval", type=int, default=None, help="Log interval.")
    parser.add_argument("--use-cuda", type=int, choices=[0, 1], default=None)
    parser.add_argument("--use-jittor-geometric", type=int, choices=[0, 1], default=None)
    parser.add_argument(
        "--export-strategy",
        choices=["auto", "best", "ensemble"],
        default=None,
        help="Export best validation seed, probability ensemble, or choose automatically.",
    )
    return parser.parse_args()


def load_config(args):
    config = dict(DEFAULT_CONFIG)
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}. "
                "Pass a valid --config path or omit --config."
            )
        with config_path.open(encoding="utf-8") as f:
            config.update(json.load(f))

    for key in config:
        arg_name = key.replace("_", "-")
        value = getattr(args, key, None)
        if value is None and hasattr(args, arg_name):
            value = getattr(args, arg_name)
        if value is not None:
            config[key] = value
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    jt.misc.set_global_seed(seed)


def parse_seeds(config):
    """Return the run seeds used for model selection and ensembling."""
    seeds = config.get("seeds")
    if seeds is None:
        return [int(config["seed"])]
    if isinstance(seeds, int):
        return [seeds]
    if isinstance(seeds, str):
        return [int(seed.strip()) for seed in seeds.split(",") if seed.strip()]
    return [int(seed) for seed in seeds]


def require_file(path, hint):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}. {hint}")


def write_run_metadata(config, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    with (output_dir / "command.txt").open("w", encoding="utf-8") as f:
        f.write(" ".join([Path(sys.executable).name] + sys.argv) + "\n")


def log_message(message, log_file):
    print(message)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


class GraphData:
    """Container for tensors needed by GCNConv."""


def load_graph(data_path):
    """Load Cora pickle and return raw dict plus Jittor graph tensors."""
    require_file(
        data_path,
        "Put cora.pkl under release/data/ or pass --data-path /path/to/cora.pkl.",
    )
    with data_path.open("rb") as f:
        raw = pickle.load(f)

    required = [
        "x",
        "y",
        "edge_index",
        "train_mask",
        "val_mask",
        "test_mask",
        "num_classes",
        "num_features",
    ]
    missing = [key for key in required if key not in raw]
    if missing:
        raise KeyError(f"{data_path} is missing required fields: {missing}")

    graph = GraphData()
    graph.x = jt.array(raw["x"].astype(np.float32))
    graph.y = jt.array(raw["y"].astype(np.int64))
    graph.edge_index = jt.array(raw["edge_index"].astype(np.int64))
    graph.train_mask = jt.array(raw["train_mask"])
    graph.val_mask = jt.array(raw["val_mask"])
    graph.test_mask = jt.array(raw["test_mask"])

    row_sum = graph.x.sum(dim=1, keepdims=True)
    row_sum = jt.clamp(row_sum, min_v=1e-12)
    graph.x = graph.x / row_sum

    v_num = graph.x.shape[0]
    edge_index, edge_weight = gcn_norm(
        graph.edge_index,
        None,
        v_num,
        improved=False,
        add_self_loops=True,
    )
    with jt.no_grad():
        graph.csc = cootocsc(edge_index, edge_weight, v_num)
        graph.csr = cootocsr(edge_index, edge_weight, v_num)

    if not USING_JITTOR_GEOMETRIC:
        dense_adj = np.zeros((v_num, v_num), dtype=np.float32)
        edge_index_np = edge_index.numpy()
        edge_weight_np = edge_weight.numpy()
        dense_adj[edge_index_np[1], edge_index_np[0]] = edge_weight_np
        graph.adj = jt.array(dense_adj)
        graph.csc = graph.adj
        graph.csr = graph.adj

    return raw, graph

# ============================================================
# 第三步：定义 GCN 模型
# ============================================================
class GCNNet(nn.Module):
    """Two-layer GCN for node classification.

    Args:
        num_features: Input node feature dimension.
        num_classes: Number of output classes.
        hidden_dim: Hidden channel size of the first GCN layer.
        dropout: Dropout probability applied between the two layers.
        graph: GraphData containing node features and sparse graph formats.
    """

    def __init__(self, num_features, num_classes, graph, hidden_dim=256, dropout=0.8):
        super(GCNNet, self).__init__()
        self.graph = graph
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels=num_features, out_channels=hidden_dim)
        self.conv2 = GCNConv(in_channels=hidden_dim, out_channels=num_classes)

    def execute(self):
        x = self.graph.x
        csc = self.graph.csc
        csr = self.graph.csr
        x = nn.relu(self.conv1(x, csc, csr))
        x = nn.dropout(x, self.dropout, is_train=self.training)
        x = self.conv2(x, csc, csr)
        return nn.log_softmax(x, dim=1)


def train(model, optimizer, graph):
    """Run one optimization step on train_mask nodes and return loss."""
    model.train()
    pred = model()[graph.train_mask]
    label = graph.y[graph.train_mask]
    loss = nn.nll_loss(pred, label)
    optimizer.step(loss)
    return loss.item()


def evaluate(model, graph):
    """Return train and validation accuracy computed from boolean masks."""
    model.eval()
    logits = model()
    accs = []

    for mask in [graph.train_mask, graph.val_mask]:
        pred, _ = jt.argmax(logits[mask], dim=1)
        correct = (pred == graph.y[mask]).sum().item()
        total = mask.sum().item()
        acc = correct / total
        accs.append(acc)

    return accs


def export_result_from_probs(probs, raw, result_path):
    """Save predictions for test_mask nodes from class probabilities."""
    pred = np.argmax(probs, axis=1)
    test_indices = np.nonzero(raw["test_mask"])[0]
    result = {str(int(idx)): int(pred[int(idx)]) for idx in test_indices}

    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def accuracy_from_probs(probs, raw, mask_name):
    mask = raw[mask_name]
    pred = np.argmax(probs[mask], axis=1)
    label = raw["y"][mask]
    return float((pred == label).sum() / mask.sum())


def package_submission(result_path, zip_path):
    """Create a competition zip containing gcn.py and result.json at root."""
    require_file(result_path, "Run training first so result.json can be generated.")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(Path(__file__).resolve(), arcname="gcn.py")
        zf.write(result_path, arcname="result.json")


def train_one_run(raw, graph, config, seed, log_file):
    """Train one seed and keep the probabilities from the best validation epoch."""
    set_seed(seed)
    model = GCNNet(
        num_features=int(raw["num_features"]),
        num_classes=int(raw["num_classes"]),
        graph=graph,
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
    )
    optimizer = nn.Adam(
        params=model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )

    best = {
        "seed": seed,
        "epoch": 0,
        "train_acc": 0.0,
        "val_acc": 0.0,
        "probs": None,
    }
    for epoch in range(1, int(config["epochs"]) + 1):
        loss = train(model, optimizer, graph)
        train_acc, val_acc = evaluate(model, graph)

        if val_acc >= best["val_acc"]:
            model.eval()
            best["epoch"] = epoch
            best["train_acc"] = train_acc
            best["val_acc"] = val_acc
            best["probs"] = np.exp(model().numpy())

        if epoch % int(config["log_interval"]) == 0 or epoch == 1:
            message = (
                f"Seed: {seed}, Epoch: {epoch:03d}, Loss: {loss:.4f}, "
                f"Train Acc: {train_acc:.4f}, Best Val Acc: {best['val_acc']:.4f}"
            )
            log_message(message, log_file)

    log_message(
        f"Seed {seed} 最佳结果: Epoch {best['epoch']}, "
        f"Train Acc: {best['train_acc']:.4f}, Val Acc: {best['val_acc']:.4f}",
        log_file,
    )
    return best


def main():
    args = parse_args()
    config = load_config(args)
    output_dir = Path(config["output_dir"])
    log_file = output_dir / "train.log"
    write_run_metadata(config, output_dir)
    log_file.write_text("", encoding="utf-8")

    jt.flags.use_cuda = int(config["use_cuda"])
    raw, graph = load_graph(Path(config["data_path"]))
    seeds = parse_seeds(config)
    log_message(f"训练 seeds: {seeds}", log_file)

    runs = [train_one_run(raw, graph, config, seed, log_file) for seed in seeds]
    ensemble_probs = np.mean([run["probs"] for run in runs], axis=0)
    ensemble_train_acc = accuracy_from_probs(ensemble_probs, raw, "train_mask")
    ensemble_val_acc = accuracy_from_probs(ensemble_probs, raw, "val_mask")
    best_single = max(runs, key=lambda run: run["val_acc"])

    log_message(
        f"最佳单模型: Seed {best_single['seed']}, Epoch {best_single['epoch']}, "
        f"Val Acc: {best_single['val_acc']:.4f}",
        log_file,
    )
    log_message(
        f"Ensemble 结果: Train Acc: {ensemble_train_acc:.4f}, "
        f"Val Acc: {ensemble_val_acc:.4f}",
        log_file,
    )

    strategy = config.get("export_strategy", "auto")
    if strategy == "best" or (
        strategy == "auto" and best_single["val_acc"] >= ensemble_val_acc
    ):
        export_probs = best_single["probs"]
        log_message(
            f"导出策略: best single seed {best_single['seed']} "
            f"(Val Acc: {best_single['val_acc']:.4f})",
            log_file,
        )
    else:
        export_probs = ensemble_probs
        log_message(
            f"导出策略: ensemble (Val Acc: {ensemble_val_acc:.4f})",
            log_file,
        )

    result_path = Path(config["result_path"])
    result = export_result_from_probs(export_probs, raw, result_path)
    package_submission(result_path, Path(config["zip_path"]))

    log_message(f"预测结果已保存到 {result_path}", log_file)
    log_message(f"共预测 {len(result)} 个测试节点", log_file)
    log_message(f"提交压缩包已保存到 {config['zip_path']}", log_file)


if __name__ == "__main__":
    main()

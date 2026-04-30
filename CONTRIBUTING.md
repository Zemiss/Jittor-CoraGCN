# Contributing

欢迎改进这个 Cora GCN 热身赛项目。为了保持结果可复现，建议每次修改都遵循下面的流程。

## Development Flow

1. 修改 `release/gcn.py` 或文档。
2. 在 `release/` 目录运行训练：

   ```bash
   JITTOR_USE_CUDA=0 python gcn.py
   ```

3. 回到仓库根目录运行提交校验：

   ```bash
   python scripts/validate_submission.py
   ```

4. 确认 `release/result.zip` 仅包含：

   ```text
   gcn.py
   result.json
   ```

## Code Style

- 保持训练逻辑清晰，不把测试集标签用于调参。
- 优先复用 Jittor / JittorGeometric API。
- 影响结果的超参数变更需要在 README 中说明。
- 不提交本地缓存、虚拟环境、IDE 配置和临时日志。

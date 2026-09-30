# 当前周期教师实现与检查工具

当前训练与测试以仓库根目录 [README](../../README.md) 为准，统一配置为 `configs/rlopsd.yaml`。

- 启动：`python scripts/train.py --help`、`python scripts/evaluate.py --help`。
- 顺序独立训练：`python scripts/train_tasks.py --root <输出目录> --tasks chemistry biology ...`。
- `launch.py`/`run_tasks.py` 为上述入口共享实现，不硬编码本机模型、数据、GPU、环境路径。
- `check_mechanism.py`：小模型6轮冻结/同步与奇数步保存恢复检查。
- `check_8b_long_memory.py --model-path <模型目录>`：独立长序列GPU显存诊断，不能替代完整分布式训练验证。
- `check_memory_fix.py --old-source <旧dp_actor.py>`：对照旧实现的数值检查，需GPU。
- 数据准备工具均显式接收 `--source` 和输出参数；生物先构造500题选择题，再由 `add_biology_true_false.py` 扩充为600题。

旧实验记录保留原始数据路径与不可变源码快照，供追溯；不要把旧启动命令当成当前入口。M=2、alpha0、恒定beta、LoRA及全有效token蒸馏仍是当前方法约束。

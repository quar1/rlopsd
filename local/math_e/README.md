# math_e：Teacher-first 经验学习与虚拟反馈

主干实现复用原 RayPPOTrainer、学生 update_policy、vLLM 和 checkpoint 管理，默认入口为仓库根目录的 `scripts/train_4b.sh`。模型下载、原始数据和正常评测入口见[仓库说明](../../README.md)。

## 入口与配置

- `scripts/train_4b.sh`：正式训练和异步评测，默认训练 GPU0、1，评测 GPU2、3。
- `scripts/train.py`：仅训练，支持 `--resume` 从 math_e 完整 checkpoint 恢复。
- `configs/math_e_qwen3_4b.yaml`：保留原 4B 的模型/数据路径、16题×8轨迹、4096回答长度、学生 lr1e-6、LoRA 和 RL+OPSD 配置。
- `local/math_e/config.yaml`：教师优化、虚拟反馈和训练探针参数。

在仓库根目录执行独立训练：

```bash
python scripts/train.py --model-path /data/models/Qwen3-4B --gpus 0,3 --run-dir /data/rlopsd-runs/RUN/train

# 恢复时使用相同的训练及方法参数，并指定新的输出目录
python scripts/train.py --model-path /data/models/Qwen3-4B --gpus 0,3 --run-dir /data/rlopsd-runs/RESUMED/train --resume /data/rlopsd-runs/RUN/train/checkpoints/global_step_50
```

通用训练参数用 `--set KEY=VALUE`，方法参数用 `--method-set KEY=VALUE`，也可用 `--method-config` 指定方法 YAML。改变探针 seed 或池大小时，另选 `--prepared-dir`；默认缓存为训练配置的 `cache_dir/math_e_v1`。恢复时必须保留原来的探针划分与顺序。

## 每轮执行顺序

1. 当前学生生成轨迹；沿用任务奖励、GRPO 优势和旧策略概率，每题全部轨迹参与学生目标。
2. 在学生更新前，对教师 FP32 LoRA 因子做 EMA（旧教师权重 0.995）。底座冻结；因子 EMA 不等同于合并后权重的 EMA。
3. 同题同时出现正确/错误回答时，以固定随机源各选一条，构造一对偏好。以本轮 EMA 起点的 log-prob 为参考计算 DPO-style 梯度；每轮只做一次教师更新，起点 margin 为 0。没有混合成功/失败的题不构造偏好。
4. 参考 CE 只将问题作为输入，将原 solution 作为目标。问题与目标整体 tokenize，监督与目标字符跨度相交的 token，不追加 EOS；先按每题 token 平均，再按题平均。
5. 教师梯度跨 rank 归约后，在全部 LoRA 参数上用 FP64 标量计算高斯几何混合（最大系数 0.3、sigma 0.5）。经验梯度缺失时回退参考梯度；全零或精确反向的退化情况跳过 Adam 与 scheduler。
6. 教师使用独立 AdamW（lr5e-6、weight decay0、clip1、无 warmup）。强度 s 缩放 EMA 和 Adam 带来的整体参数位移。s>0 提交候选 Adam moments；s=0 恢复教师全部状态，包括 EMA、moments 和 scheduler。
7. step1、11、21……串行比较 s=0 与 0.25。每支从同一学生参数、优化器、scheduler、buffers、模式、梯度及 Python/NumPy/CPU/CUDA RNG 开始，执行与真实更新相同的完整学生更新。
8. 用未参与训练更新的 8 道训练探针比较逐题参考 CE。非零强度需满足平均改善减去标准误大于 0；并列选较小强度。少于两道有效探针或所有虚拟位移为零时沿用历史强度，初始为 0.25。NaN/Inf 报错。
9. 回滚学生状态，提交教师，执行一次真实学生更新。强度保持一个 10 步窗口；非零窗口每轮重新生成教师提议，零强度窗口的非决策轮跳过提议。

学生目标仍为 RL + 0.01×全词表 KL(学生‖教师)，沿用重要性修正、token KL 裁剪、全局有效 token 平均和 EOS mask。alpha=0 取消固定 reference 正则，不取消蒸馏教师。每外层轮默认有 4 次真实学生优化器更新；虚拟更新不计入真实步数。

## 数据、日志和恢复

`prepare.py` 从原 28,519 道训练题固定留出 128 道探针题，其余 28,391 道保持原顺序；不修改源 parquet，不使用开发集或正式测试集做反馈。每次轮换 8 道探针，seed20261009。参考解从既有 solution 上下文提取。

- `resolved_config.yaml`、`settings.yaml`、`math_e_config.yaml`、`split.json`：运行配置与探针划分。
- `math_e/teacher_first.jsonl`：真实/虚拟步数、教师损失、几何统计、退化分支、位移、窗口强度和阶段资源消耗。
- `math_e/feedback_N.json`：各候选逐题 CE、平均改善、标准误、得分、选择原因、优化器步数和位移。
- `math_e/rollout_N.json`、`rollout_memory_N_PID.json`：生成种子、耗时和各 vLLM TP 进程的 PyTorch CUDA allocator 峰值。
- `checkpoints/global_step_N/actor/periodic_teacher.pt`：沿用框架文件接口，保存 math_e 版本、教师参数/Adam/scheduler/buffers、计数、探针游标和强度窗口。只支持 math_e 入口恢复。

每条生成请求的 seed 由训练 seed、外层步数、原题 index 和 rollout 序号决定，序号在分派生成 worker 前计算。虚拟分支不推进生成引擎、dataloader 或正式 checkpoint；不承诺异步 GPU 执行逐 bit 一致。

## 代码位置

| 文件 | 职责 |
|---|---|
| `launch.py`、`entry.py`、`worker.py`、`config.py` | 配置、源码快照、训练器和 worker 接入 |
| `runtime.py` | 教师更新、虚拟反馈控制、真实学生更新与恢复 |
| `mechanism.py` | 梯度几何、状态快照/回滚、候选选择和种子 |
| `prepare.py` | 训练/探针划分和参考目标 tokenization |
| `rollout.py`、`memory.py` | 生成请求种子及资源日志 |

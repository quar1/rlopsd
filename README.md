# Qwen3-4B · RL + OPSD · M2

基于 [SDPG](https://github.com/lauyikfung/SDPG) 和项目已有数学训练实现，使用后训练版本 **Qwen/Qwen3-4B**（不是 Base），训练集为 OpenThoughts 数学数据，测试为 AIME24、AIME25、HMMT February 2025。

学生、教师和测试全部使用 `enable_thinking=false`。此版本将独立教师改为 **每 2 个外层训练 step 硬复制一次学生参数**。它不是冻结初始教师组，也不是 Teacher-first 梯度更新方案。当前服务器正在运行的冻结教师实验未被本次发布修改。发布前做 CPU 机制检查和配置解析，不在占用中的 GPU 上额外启动实验；这些检查不等价于目标服务器上的完整 M2 GPU 试跑。

## 获取代码和数据

```bash
git clone https://github.com/quar1/rlopsd.git
cd rlopsd
```

仓库只包含代码和固定划分的数据，**不包含模型权重，也不需要 Git LFS**。在服务器准备好官方后训练模型 [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)，通过 `MODEL_PATH` 指定已下载的本地目录；不要改用 Qwen3-4B-Base。当前实验来源 revision 为 `1cfa9a7208912126459214e8b04321603b3df60c`。

## 环境

Linux x86_64、Python **3.10**、支持 CUDA 12.8 的 NVIDIA 驱动。沿用当前项目固定直接和间接依赖版本的 `requirements.txt`，包括与 torch 2.8 ABI 对齐的 FlashAttention wheel，无需 wandb。

```bash
conda create -n opsd python=3.10 pip -y
conda activate opsd
python -m pip install --no-cache-dir --no-compile -r requirements.txt
```

从源码启动即可，无需另外安装上游 SDPG/verl 包。源码中的修改是本项目所需的一部分。模型、数据、输出及安装临时文件应放在容量足够的数据盘；本仓库不保证根分区总占用或安装峰值低于 20 GB。

## 训练并异步测试

```bash
# 只展示实际配置，不启动 GPU 工作进程
bash scripts/train_4b.sh --dry-run

# 默认总共四卡：GPU0、1 训练，GPU2、3 分担不同 checkpoint 的异步评测
MODEL_PATH=/data/models/Qwen3-4B bash scripts/train_4b.sh

# 在其他服务器按可用卡及数据盘调整
MODEL_PATH=/data/models/Qwen3-4B TRAIN_GPUS=0,1 EVAL_GPUS=2,3 \
RUN_ROOT=/data/rlopsd-runs CACHE_DIR=/data/rlopsd-cache \
nohup bash scripts/train_4b.sh > launch.log 2>&1 &
```

`PYTHON` 可指定环境解释器；`MODEL_PATH`、`DATA_ROOT`、`RUN_ROOT`、`RUN_DIR`、`TRAIN_GPUS`、`EVAL_GPUS`、`TENSOR_PARALLEL_SIZE`、`STEPS`、`BETA`、`ROLLOUT_N`、`SAVE_EVERY`、`EVAL_EVERY`、`MAX_RESPONSE_LENGTH`、`LEARNING_RATE` 可覆盖默认值。`DATA_ROOT` 应指向包含 `math/` 的目录。完整参数见 `bash scripts/train_4b.sh --help` 和 `configs/math_m2_qwen3_4b.yaml`。

默认从当前激活环境调用 `python`，数据默认使用本仓库内文件，模型默认在仓库同级的 `models/Qwen3-4B/` 或使用 `MODEL_PATH` 指定目录，输出默认写入仓库同级的 `rlopsd-runs/`。源代码快照不会重复复制模型和数据。GPU 显存/磁盘可用量检查失败时明确退出，不会停止已有任务；本配置沿用两张 96 GB 卡的训练配置，并非已保证两张 48 GB 卡可运行。

| 项目 | 默认值 |
|---|---|
| GPU 分配 | 0、1 训练（TP2）；2、3 各一个独立评测 worker，合计4卡 |
| 模型 | Qwen/Qwen3-4B，BF16 基座，LoRA rank64 / alpha128 |
| LoRA 层 | q/k/v/o、gate/up/down projection |
| 学生 / 教师 / 测试 thinking | 全部关闭，Qwen3 原生聊天模板 |
| 训练轮数 | 200 个外层 rollout/global iteration |
| 每轮训练量 | 全局 16 题，每题 8 条，128 条回答 |
| 优化 | 每个 minibatch 4 题/32 条，4 次 Adam 更新/轮，PPO epoch1 |
| 输入 / 训练回答上限 | 4096 / 4096 tokens；thinking 关闭 |
| 学习率 | 1e-6，40 次优化器更新 warmup，之后恒定 |
| 学生目标 | RL + 0.01 × OPSD；alpha=0，KD 无 warmup/decay |
| 教师更新 | 每 2 个外层轮学生全部优化完成后硬复制 LoRA |
| 外部评测 | step0、20、40、…、200；每题 12 次，temperature1 |
| 评测回答 / 上下文上限 | 32000 / 32768；实际回答预算受 prompt 长度限制 |
| 完整 checkpoint | step50、100、150、200，最多保留 4 份 |
| 中间评测快照 | step20、40、60、80、120、140、160、180，只保存学生 LoRA |
| 每步日志 | RL/OPSD 损失；前 4 个题组各 8 条，共 32 条学生轨迹 |

两张评测卡通过文件锁领取不同 checkpoint，避免重复测试同一节点；每个 worker 自己完成该节点的三套题。异步评测有积压时按队列继续，**每 20 步生成评测任务，不保证每 20 步的墙钟时间内评测完成**。训练结束后控制器等待所需评测完成。

## M2 的训练顺序

1. 学生只读取题目，生成当前批次轨迹；数学正确性奖励为 1，否则为 0。
2. 保存旧策略概率，按同题 8 条回答计算 GRPO 优势。
3. 用当前独立、冻结教师在相同学生回答前缀上计算完整词表分布。教师额外看到原数据 `solution` 字段中的参考解题过程；学生输入不含该字段。
4. 完成本轮所有学生 minibatch、梯度累积和优化器更新。教师在此期间不变。
5. 若完成的外层轮数为 2 的倍数，将本轮更新后的学生 LoRA 硬复制给教师，供下一轮使用。

第 1、2 轮使用初始教师；第 3、4 轮使用第 2 轮结束时的学生快照。教师没有优化器，没有教师 CE、bootstrap、EMA、高斯更新或虚拟学生。

OPSD 是全词表 KL(学生 || 教师)，教师分布停止梯度，所有有效 response token 参与，包括零/负优势回答及有效 EOS。保留 PPO/DAPO 裁剪、旧策略概率、重要性修正、token KL 裁剪及全局有效 token 平均。alpha=0 取消固定 reference 正则和无用 reference worker，不取消 rollout 旧策略概率或蒸馏教师。

实现：`verl/workers/actor/periodic_teacher.py`、`verl/workers/fsdp_workers.py::update_actor`。教师保存独立参数，记录参数摘要并检查整轮未变；同步事件和最近同步步写入日志及 checkpoint，恢复时核验周期。

## 数据

`data/math/` 已包含固定划分的训练、开发和测试文件，详见 [data/README.md](data/README.md)。不会在新服务器重新抽样划分。

- `train_teacher.parquet`：28,519 条训练题，含教师参考解上下文。
- `train_plain.parquet`：同一批训练题的纯学生输入。
- `test.parquet`：256 条开发题，**不是**三套正式测试；默认内置开发评测关闭。
- `benchmarks/{aime24,aime25,hmmt25}.parquet`：每套 30 题。
- `split_ids.json`、`manifest.json`、`rejected.json`：固定 ID、来源、处理规则和过滤记录。

## 单独测试与恢复

联合训练入口已自动启动评测，不要对同一输出目录重复启动第二个评测器。若需要单独评测：

```bash
# 初始模型
MODEL_PATH=/data/models/Qwen3-4B EVAL_GPUS=2 bash scripts/eval_4b.sh
# 完整 checkpoint；也支持完整发布的 eval_snapshots/global_step_N
MODEL_PATH=/data/models/Qwen3-4B EVAL_GPUS=2 bash scripts/eval_4b.sh \
  --checkpoint /data/rlopsd-runs/RUN/train/checkpoints/global_step_50
# 持续监控已有训练，评测 step0/20/.../200
MODEL_PATH=/data/models/Qwen3-4B EVAL_GPUS=2 bash scripts/eval_4b.sh \
  --watch-run /data/rlopsd-runs/RUN/train --include-initial --every 20 --last-step 200
```

完整断点恢复使用底层训练入口，指定新输出目录及原完整 checkpoint（中间 LoRA 快照不可恢复训练）：

```bash
python scripts/train.py --task math --config configs/math_m2_qwen3_4b.yaml \
  --model-path /data/models/Qwen3-4B \
  --run-dir /data/rlopsd-runs/RESUMED/train \
  --resume /data/rlopsd-runs/RUN/train/checkpoints/global_step_50
```

从仓库根目录执行恢复命令；该入口只恢复训练，若要异步评测，另外启动 watcher。保存内容包括学生、独立教师、学生优化器、scheduler、RNG、数据位置、外层步数与最近教师同步步数。不能将冻结教师 checkpoint 直接当成 M2 checkpoint 恢复，机制会检查 interval。

## 输出

- `train/losses.csv`：每步 RL、原始 KL、加权/未加权 OPSD、总损失、奖励、长度与截断率；beta=0.01。
- `train/metrics.jsonl`：完整训练指标，含教师同步事件数值。
- `train/rollouts/<step>.jsonl`：每步选取前 4 个题组的完整学生回答和 token ID；属于固定选择的诊断样本。
- `train/train.log`：训练日志及 `PERIODIC_TEACHER` 事件。
- `evaluation/step_XXXX/`：三套测试的每条生成、判分和 Avg@12 / Pass@12 汇总。
- `train/checkpoints/`：可恢复断点；`train/eval_snapshots/`：仅推理快照。
- `launch.json`、`source/configs/shell_launch.yaml`：启动配置；`source/`：不可变源码快照。

CPU 回归检查：`PYTHONPATH=. CUDA_VISIBLE_DEVICES='' python tests/check_m2_contract.py`。它检查 2/4/6 轮同步、教师独立和冻结、负/零优势蒸馏、padding mask、alpha0 去 reference 和奇数轮恢复；不占用训练 GPU。

## 来源与许可

代码基于 SDPG/verl，保留 Apache-2.0 许可和 Notice；上游版本记录在 `DOWNLOAD_RECORD.json`。模型不在本仓库内分发；其许可参见官方模型仓库。数据来源及相应上游页面见 `data/README.md`；各项资源按其各自许可使用。

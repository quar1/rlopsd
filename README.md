# RL+OPSD：固定周期教师更新的科学问答训练

本项目基于 [SDPG 公开实现](https://github.com/lauyikfung/SDPG)（上游快照 `b34e7b78c748a18b58a4b73c4d2743db0c397c78`）和 verl，研究学生使用 GRPO 强化学习与 on-policy 蒸馏联合训练时的周期教师基线。默认使用 **Qwen3-8B-Base、LoRA、固定每2个外层 step 硬同步教师**。

科学任务来源于 SciKnowEval L3。任务选择和部分超参数参考 [When Should the Teacher Move? Temporal Coupling and Stability in Self On-Policy Distillation](https://arxiv.org/abs/2606.03532)。本项目使用独立冻结教师、答案特权上下文、全词表反向 KL 和 LoRA；不是原论文 CGTR 或全部训练细节的严格复现。当前入口不加入 Teacher-first 学习分支、教师梯度/CE、EMA、高斯正交、bootstrap 或元梯度。

## 目录

```text
configs/rlopsd.yaml          当前实验默认参数
scripts/train.py            单任务训练、恢复训练
scripts/evaluate.py         单独评估初始模型或完整 checkpoint
scripts/train_tasks.py      多任务顺序训练，每科独立初始化
local/periodic_m2/           配置构建、数据准备和机制检查
local/when_chemistry/        共享数据转换与四科选项奖励（历史目录名）
verl/                       SDPG/verl 训练核心及周期教师实现
data/                      随仓库携带的四科固定数据
requirements.txt            Python 3.10、CUDA 12.8 运行依赖锁定
LICENSE / Notice.txt        上游许可证与声明
```

`scripts/` 只保留当前训练、测试入口。已移除上游示例、通用文档、Docker配置、项目网页、论文图表及通用测试；保留 `verl/` 训练框架、当前方法检查工具、数据准备代码与上游许可证。运行依赖统一使用 `requirements.txt`，模型权重、Conda环境或checkpoint无需进入Git仓库。

## 环境

```bash
conda create -n opsd python=3.10 pip -y
conda activate opsd
python -m pip install --no-cache-dir --no-compile -r requirements.txt
python -m pip install --no-deps -e .
```

Linux x86_64、NVIDIA GPU，依赖锁定为 PyTorch 2.8.0/CUDA12.8、vLLM0.11.0、Transformers4.57.1等，具体以 requirements.txt 为准。启动脚本会设置仓库的 PYTHONPATH。使用 console/file 日志，不要求 wandb。

## 数据集

默认使用仓库 `data/`。`--data-root` 可以指定外部数据根目录，各任务目录名称如下：

| 目录/任务 | 训练 | 测试 | 内容 |
|---|---:|---:|---|
| chemistry | 1890 | 210 | L3四选一 |
| biology | 540 | 60 | 300道四选一、200道二选一、100道判断题 |
| physics | 720 | 80 | L3四选一 |
| materials | 839 | 94 | L3四选一，剔除1道格式异常题、1道重复题 |

每科包含：

- `train_teacher.parquet`：当前训练实际读取的文件，含教师专用正确选项提示。
- `train_plain.parquet`：相同训练题，不带特权提示；用于核对和其他基线。
- `test.parquet`：测试题，输入不包含正确答案提示；标签仅供评分。
- `manifest.json`：来源、划分信息、ID或原始清单、文件哈希。

`train_teacher` 和 `train_plain` 是同一批题的两种表示，不应拼接扩充训练集。原始数据来自 [SciKnowEval](https://huggingface.co/datasets/hicai-zju/SciKnowEval)，项目副本数据卡标注 MIT；分发时保留来源和许可证信息。清单中的原始绝对路径是数据来源记录，不是运行时依赖。

四科均检查训练/测试 ID 和归一化题目无交叉。选择题保留既有seed42划分；生物新增判断题单独seed42划分90/10，保持原450/50选择题归属不变。判断题读取原始 `answer` 中的 Yes/No，映射为 `A. Yes / B. No`。材料异常与重复原记录单独保存。具体见 `data/SCIENCE_DATASETS.md`。

**这些是本地固定划分，不声称与 When 的作者题目ID完全一致。生物540/60和材料839/94也与论文数量不同。** 生物应分别报告选择题与判断题结果，逐题输出的元数据保留题型；当前默认汇总仍是全测试集准确率。化学测试集在本项目历史实验中多次使用，不应当作从未接触的最终确认集。

## 默认配置

所有外层 step 均指一批 rollout 加本轮全部学生优化，不是 micro-batch、梯度累积次数或 token 数。

| 参数 | 默认值 |
|---|---|
| 模型 | Qwen/Qwen3-8B-Base（预训练基座） |
| LoRA | rank64、alpha128、all-linear；BF16基座、FP32 adapter |
| GPU | 新服务器默认 `0,1,2,3` 四卡；当前旧服务器 `configs/server.yaml` 使用 `4,5` 双卡 |
| 总步数 | 300 |
| 题目batch / 每题rollout | 16 / 4，每轮64条回答 |
| PPO mini-batch | 4道题的数量尺度，对应16条回答 |
| 每卡micro-batch | 1条；四卡时每卡累积4条后更新（双卡配置为8条） |
| PPO epochs | 1；每轮4次学生optimizer.step |
| Prompt / response上限 | 2048 / 8192 tokens，thinking开启 |
| 训练采样 | temperature1、top_p1、top_k=-1 |
| 优化器 | AdamW，lr1e-5，warmup10个外层step，之后恒定 |
| weight decay / grad clip | 0.01 / 1.0 |
| 教师更新 | 固定M=2，完成偶数外层step后硬复制 |
| 损失 | RL + 0.01 × OPSD；固定reference正则alpha=0 |
| 测试 | 初始一次，之后每10步及最终步；greedy、每题1条 |
| 保存 | 化学每5步，其余三科每50步；最终步必存；最多保留最近2个完整恢复点 |
| vLLM | 四卡组成2个TP2生成副本；每副本max_num_seqs16、每卡显存利用率0.45 |

`configs/rlopsd.yaml` 保存完整参数。可以用 `--config` 指定配置副本，常用参数直接用命令行覆盖，其他参数使用重复的 `--set KEY=VALUE`。优先级：默认配置 → 配置文件 → 对应task_overrides → smoke配置 → --set →专用命令行参数。

当前固定周期入口有意要求 `teacher_update_interval=2`、LoRA、alpha0、恒定beta及全有效token蒸馏，避免误切回上游即时教师算法。测试/保存频率可配置为其他正整数，设为-1关闭周期事件；初始测试由 `val_before_train` 单独控制。未实现根据验证集自动选择最佳模型或准确率下降自动停止。

## 新服务器：默认四卡

通用配置 `configs/rlopsd.yaml` 使用物理GPU `0,1,2,3`：学生训练使用4卡FSDP，生成使用2个独立vLLM副本，每个副本TP2。4个agent worker负责调度请求；两个生成副本分担同一批64条回答，不将rollout总量翻倍。

全局训练设置保持16题×4回答=64条/step，每个mini-batch16条回答，分到4卡各4条；每卡micro-batch1、累积4次后更新，每个外层step仍有4次学生优化。教师每2个外层step同步，分科测试/保存频率不随GPU数量变化。

```bash
python scripts/train_tasks.py \
  --model-path /data1/rlopsd/models/Qwen3-8B-Base \
  --gpus 0,1,2,3 \
  --set output_root=/data1/rlopsd/runs \
  --set cache_dir=/data1/rlopsd/cache --set offline=true
```

上述模型路径应替换为新服务器实际路径。默认读取仓库 `data/`，也可传 `--data-root`。先加 `--render-only` 检查配置。模型分片、教师前向与全词表激活有不同的显存开销，四卡不意味着所有单卡峰值都减半；实际显存余量需结合新卡型号和容量验证。

## 当前旧服务器配置与分科频率

本机使用 `--config configs/server.yaml`：已下载Base模型位于 `/data1/wcy/opsd/rl/Qwen3-8B-Base`；训练输出和checkpoint默认写到 `/data1/wcy/opsd/rlopsd/runs/`。不额外复制模型权重。省略 `--run-dir` 时自动生成带时间戳的任务目录，checkpoint在其中 `checkpoints/global_step_N/actor/`。

| 任务 | 完整测试频率 | checkpoint频率 |
|---|---:|---:|
| chemistry | 每10步 | 每5步 |
| biology | 每10步 | 每50步 |
| physics | 每10步 | 每50步 |
| materials | 每10步 | 每50步 |

初始测试和最终测试仍开启；最终步也保存。化学第5步保存的是奇数步状态：教师最近同步为第4步，恢复时保留该状态，第6步学生优化后再同步。各科checkpoint仍最多保留最近2个。

```bash
python scripts/train.py --config configs/server.yaml --task chemistry
python scripts/train_tasks.py --config configs/server.yaml
```

第二条顺序独立运行四科，自动生成 `runs/suite-<时间戳>/`。示例命令不会因文档更新而自动执行。跨机器使用通用 `configs/rlopsd.yaml` 并覆盖模型与输出位置。`thinking=true`仍传给模板，但Base未经过指令/推理后训练，不保证它会遵守推理与答案标签格式。

## 启动训练

以下从仓库根目录执行，模型位置按实际机器设置。模型参数也可用 Hugging Face 模型ID（需要网络和下载空间）。

```bash
python scripts/train.py \
  --task chemistry \
  --model-path /data1/wcy/opsd/rl/Qwen3-8B-Base \
  --data-root /data1/wcy/opsd/rlopsd \
  --gpus 4,5 \
  --run-dir /data1/wcy/opsd/rlopsd/runs/chemistry-new-run \
  --beta 0.01 --steps 300 --test-freq 10 --save-freq 5 \
  --set offline=true
```

省略 `--data-root` 即读取仓库内相同的固定数据。更改batch、长度、学习率、缓存目录等示例：

```bash
python scripts/train.py --task biology --model-path /path/to/Qwen3-8B-Base \
  --gpus 0,1,2,3 --run-dir /data/runs/biology \
  --set train_batch_size=16 --set rollout_n=4 \
  --set max_response_length=8192 --set learning_rate=0.00001 \
  --set cache_dir=/data/cache --set ray_temp_root=/tmp
```

加 `--render-only` 只检查并生成完整配置，不启动GPU worker。加 `--smoke` 则使用3步、4题、response128和4题测试的小检查，成绩不作正式比较。

连续运行多个任务：

```bash
python scripts/train_tasks.py \
  --root /data/runs/science-suite --model-path /path/to/Qwen3-8B-Base \
  --gpus 0,1,2,3 --set offline=true
```

默认依次运行 chemistry → biology → physics → materials，无需指定 `--tasks`。每个任务从同一初始模型独立开始，不接着前一科的权重训练；正常完成后自动启动下一科，失败停止队列。需要运行部分任务时才传入例如 `--tasks chemistry biology`。

## 恢复训练与单独测试

恢复必须指定完整 `checkpoints/global_step_N/`，同样的初始基座、LoRA结构及兼容GPU布局。使用新的输出目录保留原日志。`--steps` 表示恢复后的总目标step，不是额外步数。

```bash
python scripts/train.py --task chemistry --model-path /path/to/Qwen3-8B-Base \
  --gpus 0,1,2,3 --run-dir /data/runs/chemistry-resume \
  --resume /data/runs/chemistry/checkpoints/global_step_20 --steps 300
```

单独评估已有完整checkpoint：

```bash
python scripts/evaluate.py --task chemistry --model-path /path/to/Qwen3-8B-Base \
  --gpus 0,1,2,3 --run-dir /data/eval/chemistry-step20 \
  --checkpoint /data/runs/chemistry/checkpoints/global_step_20
```

省略 `--checkpoint` 评估初始模型。评估复用相同训练框架加载完整状态，随后 `val_only=True` 执行测试并返回，不采集训练rollout或更新参数；因此仍需同科训练数据和框架初始化资源，不是轻量独立推理服务。评估总会开启进入循环前的测试，即使提供checkpoint也不会误进入训练。支持 `--set eval_do_sample=true --set eval_temperature=1.0 --set eval_n=4` 调整评估采样。

新运行记录数据指纹；恢复时如果找到原运行的指纹且不一致会拒绝，避免换了划分却声称精确续训。旧实验可能没有指纹文件，需人工核对其manifest。仅adapter目录不能作为 `--checkpoint`。

## 每轮训练的实际顺序

1. 初始化学生LoRA和独立冻结教师，复制初始adapter；教师无梯度、无优化器。首次正式训练先评估初始学生。
2. 从训练集抽取16题。数据加载器拆分 `[TEACHER_CONTEXT_TOKEN]`：学生仅见题目和选项；教师额外见正确选项提示。学生每题采样4条回答。
3. 按最后一个完整 `<answer>...</answer>` 标签评分：选项正确1分，否则0；解析失败0。判断题也输出A/B。组内4条回答计算GRPO标准化优势；保留旧策略log-prob和有效response mask。
4. 整轮教师参数冻结。在每个micro-batch中，用该教师在学生已生成的前缀上计算分布，再计算学生分布和全词表KL(student‖teacher)。教师分布stop-gradient，教师没有自己的轨迹采样。
5. 学生优化 `L_RL + beta × L_OPSD`。PPO/DAPO上下裁剪0.2/0.28、dual-clip10、重要性比率和token-mean聚合沿用SDPG；每token蒸馏KL上限20。负优势、零优势及EOS的有效response token仍参加KD；padding不参加。alpha0不创建固定reference worker，但保留旧策略概率及独立蒸馏教师。
6. 本轮所有mini-batch更新完成后，检查教师参数在轮内未变化。仅第2/4/6…步硬复制更新后的学生adapter给教师。第1/2轮用theta0，第3/4轮用theta2，依此类推。教师前向按需上GPU，结束后卸载CPU；独立参数不会隐式共享更新。
7. 到保存步则写checkpoint，再同步学生adapter到vLLM；到测试步使用更新后的学生对完整测试集生成并评分。每个完成step记录训练指标，进入下一批在线采样。

教师每轮做冻结/梯度/步数一致性检查；reward、生成长度及p95/p99、熵、RL/KD损失、同步事件和耗时每轮记录。**每10步的检测指完整测试集评估，不是教师是否更新的门控判断。** 这里没有CGTR式门控或额外固定频率崩溃判定器。

## 输出与当前运行限制

每个任务输出目录保存 `launch.json`（精确命令、环境与教师提示）、`settings.yaml`、`resolved_config.yaml`、`data_manifest.json`、`data_fingerprint.json`、`status.json`、`train.log`（单独测试为`evaluate.log`）、`metrics.jsonl`、`evaluation/`、`rollouts/`、`checkpoints/`。

checkpoint包含学生参数、学生优化器、scheduler/RNG、dataloader及独立教师adapter、外层完成步数和最近同步步数。冻结基座由相同初始模型/学生checkpoint确定，不重复保存一份教师基座。不能用其他任务或其他划分的checkpoint接续训练。

显存优化保留了教师前向后立即CPU卸载及全词表KL沿token维256分块checkpoint。vLLM sleep使用level1以保留冻结基座。

源码遵循上游LICENSE及Notice.txt；SciKnowEval按数据来源声明使用。新配置和数据路径迁移不更改已有运行的不可变源码快照和日志。

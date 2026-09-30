# SDPG M2 环境独立文档调用核验

核验时间：2026-09-30 04:12–04:13 +08:00。由 fresh agent 按 `run-experiment` 的 `compute-env-contract` 执行。本核验 agent 未修改代码、安装依赖或启动正式训练；只运行指定 smoke 脚本与只读诊断。`verify.py` 自身包含微型模型的一次 LoRA/Adam 更新。父 agent 修正 SDPG 检查脚本后，本 agent 按相同调用复验，保留初次失败记录。

## 结论

- **指定 L40S GPU4 上 GPU kernel / tiny LoRA：PASS，退出码 0。** 使用已确认 UUID 绑定；原数字绑定命令实际上命中 Blackwell，不能作为指定 GPU4 核验。
- **SDPG import/config smoke：最终 PASS，退出码 0。** 初次退出码 1；父 agent 修正 torch 元数据后缀断言后，同一命令通过模块导入、数学评分与 Hydra 展开。
- **已通过本轮 kernel 与 import/config smoke 范围的验证。** 未验证真实 8B 模型加载、vLLM rollout、多 GPU NCCL 或正式训练；资源与模型 ledger 差异仍应更新。

## 读取的依据

- `/home/wangchenyu/.codex/skills/run-experiment/SKILL.md`
- `/home/wangchenyu/.codex/skills/shared-references/compute-env-contract.md`
- `/home/wangchenyu/OPSD/.aris/compute/local-sdpg-m2.json`
- `/home/wangchenyu/OPSD/.aris/compute/local.md`
- `envs/opsd-cu128/README.md`、`verify.py`、`verify-sdpg.py`

`pyvenv.cfg` 确认 `home = /data1/envs/opsd/bin`、`include-system-site-packages = true`、Python 3.10.20，符合隔离 overlay 使用共享基底包的说明。项目根目录未发现 `AGENTS.md`。

## 逐字执行与结果

### 1. 原指定命令：退出码 0，但 GPU 绑定假设不成立

```bash
cd /home/wangchenyu/OPSD && CUDA_VISIBLE_DEVICES=4 /data1/wcy/opsd/envs/sdpg-m2/bin/python envs/opsd-cu128/verify.py
```

```json
{
  "python": "3.10.20",
  "torch": "2.8.0+cu128",
  "cuda": "12.8",
  "transformers": "4.57.1",
  "peft": "0.17.1",
  "flash_attn": "2.8.3",
  "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
  "loss": 4.828607082366943,
  "kernel_and_tiny_lora": "PASS"
}
```

命令在收到父 agent 关于枚举差异的更正前已启动，完成后没有重复数字绑定。实际 CUDA 默认编号不同于 `nvidia-smi` 索引，数字 `4` 未选择目标 L40S。仅凭输出型号无法确定此次命中的具体 Blackwell UUID。

### 2. 原指定 SDPG 命令：退出码 1

工作目录 `/home/wangchenyu/OPSD`：

```bash
/data1/wcy/opsd/envs/sdpg-m2/bin/python envs/opsd-cu128/verify-sdpg.py /home/wangchenyu/OPSD/dualopsd/rlopsd
```

完整错误：

```text
Traceback (most recent call last):
  File "/home/wangchenyu/OPSD/envs/opsd-cu128/verify-sdpg.py", line 14, in <module>
    assert versions['torch']=='2.8.0+cu128'
AssertionError
```

本次初始失败前已执行 `import verl`、本地 fork 路径断言、包元数据读取及 NumPy 1.26.4 断言。该次失败后没有执行主要模块导入、math-verify 样例或 Hydra `--cfg job`；后续复验见第 4 项。

只读诊断显示 `importlib.metadata.version('torch') == '2.8.0'`，元数据目录为 `/data1/envs/opsd/lib/python3.10/site-packages/torch-2.8.0.dist-info`，而实际模块报告 `torch.__version__ == '2.8.0+cu128'`、`torch.version.cuda == '12.8'`。因此本次直接失败原因是脚本要求 distribution metadata 必须包含本地 CUDA 后缀；它不是一次已证实的 CUDA kernel 故障。未绕过或修复该断言。

其余读取到的包元数据：

| 包 | 版本 |
|---|---|
| numpy | 1.26.4 |
| transformers | 4.57.1 |
| vllm | 0.11.0 |
| ray | 2.48.0 |
| torchdata | 0.11.0 |
| tensordict | 0.10.0 |
| hydra-core | 1.3.2 |
| antlr4-python3-runtime | 4.9.3 |
| math-verify | 0.8.0 |

### 3. 更正后的 GPU4 命令：退出码 0

父 agent 提供并明确要求执行的新文档调用：

```bash
cd /home/wangchenyu/OPSD && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-18ea7690-4b23-bbfd-b455-f6efe865ab85 /data1/wcy/opsd/envs/sdpg-m2/bin/python envs/opsd-cu128/verify.py
```

返回版本与第 1 项相同，`gpu = NVIDIA L40S`、`loss = 4.828607082366943`、`kernel_and_tiny_lora = PASS`。

脚本 seed=42，检查 CUDA / BF16 可用，实际执行 BF16 causal FlashAttention 前向及反向并断言输出/梯度有限；构建微型 Qwen3（2 层、hidden size 32），安装 q_proj/v_proj LoRA，执行 loss.backward 与 AdamW.step，检查 loss/梯度有限，最后 CUDA 同步。没有下载或加载 8B 权重。

两次 `verify.py` 均出现同一非致命警告：`You are attempting to use Flash Attention 2 without specifying a torch dtype. This might lead to unexpected behaviour`。脚本随后把微型模型转换至 CUDA BF16，所有断言通过；警告仍应如实保留。

### 4. 父 agent 修正检查脚本后，原 SDPG 命令复验：退出码 0

再次逐字执行第 2 项同一命令。检查脚本现允许 metadata 的基础版本为 2.8.0，同时严格断言 `torch.__version__ == '2.8.0+cu128'` 与 `torch.version.cuda == '12.8'`。本 agent 没有修改该脚本。

```json
{
  "versions": {
    "torch": "2.8.0",
    "numpy": "1.26.4",
    "transformers": "4.57.1",
    "vllm": "0.11.0",
    "ray": "2.48.0",
    "torchdata": "0.11.0",
    "tensordict": "0.10.0",
    "hydra-core": "1.3.2",
    "antlr4-python3-runtime": "4.9.3",
    "math-verify": "0.8.0"
  },
  "verl_path": "/home/wangchenyu/OPSD/dualopsd/rlopsd/verl/__init__.py",
  "imports_and_config": "PASS",
  "gpu_and_distributed_training": "NOT_TESTED"
}
```

通过核验的模块为 `verl.trainer.main_ppo`、`verl.workers.actor.dp_actor`、`verl.workers.rollout.vllm_rollout.vllm_async_server`；math-verify 成功验证相等的 boxed 2；Hydra `--cfg job` 子进程退出成功且配置包含 `actor_rollout_ref`。没有启动 Ray 集群。

vLLM 导入输出混合设备警告并明确提示设置 `CUDA_DEVICE_ORDER=PCI_BUS_ID`，进一步支持前述资源绑定差异。父 agent 另报告 overlay `pip check` 全部通过；该结果不是本 agent 独立执行的验证，因此不列入本次独立通过项。

## 文档/任务差异

1. `local-sdpg-m2.json` 的 `gpu_indices` 仍为 `[2, 4]`；当前任务已指定 `nvidia-smi` **GPU4/5**，该字段陈旧。本 agent 没有修改它。
2. 原文 `CUDA_VISIBLE_DEVICES=4` 不足以指向 `nvidia-smi` GPU4。只读 `nvidia-smi` 确认：GPU4 是 L40S，UUID `GPU-18ea7690-4b23-bbfd-b455-f6efe865ab85`，PCI `00000000:B1:00.0`；GPU5 是 L40S，UUID `GPU-ed405bf3-4069-2d20-0432-9ecda26de366`，PCI `00000000:DD:00.0`。本次只核验了更正绑定后的 GPU4，没有核验 GPU5。
3. JSON 未声明模型路径；现有 `local.md` 是 2026-09-28 的基础 opsd ledger，模型仍写 `/data1/wcy/opsd/rl/Qwen3-8B-Base`，不能代替本次 **`/data1/hf-models/Qwen3-8B`** 的 SDPG 环境/权重记录。两个 smoke 脚本均不校验该实际模型目录。
4. 初始 `verify-sdpg.py` 对 torch distribution metadata 的完整字符串断言与共享基底实际元数据不符；父 agent 已修正，复验通过。该差异已解决。
5. 本次读取的 `local.md` 未包含 sdpg-m2 的新 hash/validation ledger 区块；该环境不可借用旧 opsd 的通过记录宣称已通过本轮验证。

本核验 agent 只报告差异和复验结果，没有修复代码或 ledger，没有终止任何其他进程。

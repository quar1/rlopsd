#!/usr/bin/env bash
# Qwen/Qwen3-4B math_e: GPU0/1 training, GPU2/3 asynchronous evaluation by default.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'HELP'
Usage: bash scripts/train_4b.sh [--dry-run]

Launch fresh OpenThoughts RL+OPSD training and asynchronous AIME24/25/HMMT25 evaluation.
Student, teacher, and evaluation all disable thinking; teacher updates FIRST; virtual feedback chooses strength at steps 1,11,21,...
Defaults: 200 steps, beta=0.01, evaluate every20, 12 samples/test question.
Full checkpoints every50; evaluation every20 uses adapter exports between checkpoints.
Defaults: GPU0,1 train, GPU2,3 eval; datasets are bundled; set MODEL_PATH to an external Qwen3-4B model directory.

Environment overrides (set before bash):
  PYTHON                  Interpreter in an installed RL+OPSD environment
  MODEL_PATH, DATA_ROOT   Model directory; data root containing math/manifest.json
  CONFIG_FILE             Training YAML, relative to rlopsd/ or absolute
  TRAIN_GPUS, EVAL_GPUS   Disjoint physical GPU indices, comma-separated
  TENSOR_PARALLEL_SIZE    Training rollout TP (default2)
  STEPS, BETA             Defaults: 200,0.01
  SAVE_EVERY, EVAL_EVERY  Full-checkpoint / evaluation intervals (50/20)
  ROLLOUT_N               Answers per training question (default8)
  MAX_RESPONSE_LENGTH     Training answer token limit (default4096, thinking disabled)
  LEARNING_RATE           Defaults to selected YAML
  RUN_ROOT, RUN_DIR       Output parent or exact fresh run directory, outside source
  CACHE_DIR, RAY_TEMP_ROOT  Override YAML machine-specific working directories
  EVAL_MEMORY_UTILIZATION, EVAL_MAX_NUM_SEQS  Defaults0.8/8
  WAIT_FOR_TRAIN_MEMORY   Optional1: start evaluation, queue training for resources
  TRAIN_MIN_FREE_GIB      Optional free-memory threshold/card; default resource checks apply
  CHECKPOINT_BUDGET_GIB   Per-checkpoint capacity estimate (default12 GiB)

--dry-run prints settings/command without writing files or launching GPU workers.
Normal launch saves an immutable source snapshot, configuration and launch command.
Logs: RUN_DIR/train/train.log and RUN_DIR/evaluate_gpuN.log.
Checkpoints: RUN_DIR/train/checkpoints/global_step_N; results: RUN_DIR/evaluation/.
The launcher starts the main training directly after device/storage checks.
To detach: nohup bash scripts/train_4b.sh > launch.log 2>&1 &
HELP
    exit 0
fi
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--dry-run" ) ]]; then
    echo 'Only --help or --dry-run is accepted; configure through environment variables.' >&2
    exit 2
fi
PYTHON="${PYTHON:-python}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
exec "$PYTHON" - "$PROJECT_ROOT" "${1:-}" <<'PY'
import datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import yaml

source = Path(sys.argv[1]).resolve()
env = os.environ
config_path = Path(env.get('CONFIG_FILE', 'configs/math_e_qwen3_4b.yaml')).expanduser()
if not config_path.is_absolute():
    config_path = source/config_path
settings = yaml.safe_load(config_path.read_text())
settings.update(
    model_path=str(Path(env.get('MODEL_PATH', str(source.parent/'models/Qwen3-4B'))).expanduser().resolve()),
    data_root=str(Path(env.get('DATA_ROOT', str(source/'data'))).expanduser().resolve()),
    gpus=env.get('TRAIN_GPUS', settings['gpus']),
    tensor_parallel_size=int(env.get('TENSOR_PARALLEL_SIZE', settings['tensor_parallel_size'])),
    steps=int(env.get('STEPS', 200)),
    save_freq=int(env.get('SAVE_EVERY', env.get('EVERY', settings.get('save_freq', 20)))),
    eval_export_freq=int(env.get('EVAL_EVERY', settings.get('eval_export_freq', 0))),
    beta=float(env.get('BETA', 0.01)),
    rollout_n=int(env.get('ROLLOUT_N', settings['rollout_n'])),
    max_response_length=int(env.get('MAX_RESPONSE_LENGTH', settings['max_response_length'])),
    thinking=False, teacher_thinking=False, math_prompt_format='native',
    teacher_update_interval=-1, test_freq=-1, val_before_train=False,
)
settings['cache_dir'] = str(source.parent/'rlopsd-cache')
settings['ray_temp_root'] = '/tmp'
settings['output_root'] = str(source.parent/'rlopsd-runs')
for key, variable in [('learning_rate', 'LEARNING_RATE'), ('cache_dir', 'CACHE_DIR'),
                      ('ray_temp_root', 'RAY_TEMP_ROOT')]:
    if variable in env:
        settings[key] = float(env[variable]) if key == 'learning_rate' else env[variable]
if settings['steps'] <= 0 or settings['save_freq'] <= 0 or settings['steps'] % settings['save_freq']:
    raise SystemExit('STEPS and EVERY must be positive, with STEPS divisible by EVERY.')
if settings['eval_export_freq'] < 0 or (settings['eval_export_freq'] and settings['steps'] % settings['eval_export_freq']):
    raise SystemExit('EVAL_EVERY must be nonnegative and divide STEPS when enabled.')
settings['max_checkpoints'] = settings['steps']//settings['save_freq']
train_gpus = settings['gpus'].split(',')
eval_gpus = env.get('EVAL_GPUS', '2,3').split(',')
if (not all(g.isdigit() for g in train_gpus+eval_gpus)
        or len(set(train_gpus)) != len(train_gpus)
        or len(set(eval_gpus)) != len(eval_gpus) or set(train_gpus) & set(eval_gpus)):
    raise SystemExit('TRAIN_GPUS and EVAL_GPUS must contain distinct, disjoint numeric indices.')
if settings['tensor_parallel_size'] <= 0 or len(train_gpus) % settings['tensor_parallel_size']:
    raise SystemExit('Training GPU count must be divisible by TENSOR_PARALLEL_SIZE.')
stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
parent = Path(env.get('RUN_ROOT', settings['output_root'])).expanduser().resolve()
model_name = Path(settings['model_path']).name.lower()
root = Path(env.get('RUN_DIR', str(parent/f'math-{model_name}-nonthinking-math-e-{settings["steps"]}-{stamp}'))).expanduser().resolve()
if root == source or source in root.parents:
    raise SystemExit('RUN_DIR must be outside rlopsd/ to avoid recursive source copying.')
if root.exists():
    raise SystemExit(f'RUN_DIR already exists; choose a fresh directory: {root}')
settings['output_root'] = str(parent)
snapshot = root/'source'
command = [sys.executable, '-m', 'local.math_e.run_async', '--run-dir', str(root),
    '--config', 'configs/shell_launch.yaml',
    '--eval-gpus', ','.join(eval_gpus),
    '--eval-memory-utilization', env.get('EVAL_MEMORY_UTILIZATION', '0.8'),
    '--eval-max-num-seqs', env.get('EVAL_MAX_NUM_SEQS', '8')]
env.setdefault('CHECKPOINT_BUDGET_GIB', '12')
if env.get('WAIT_FOR_TRAIN_MEMORY', '0') == '1':
    command.append('--wait-for-train-memory')
for variable, flag in [('TRAIN_MIN_FREE_GIB', '--train-min-free-gib'),
                       ('CHECKPOINT_BUDGET_GIB', '--checkpoint-budget-gib')]:
    if variable in env:
        command.extend([flag, env[variable]])
record = dict(run_dir=str(root), config_source=str(config_path), settings=settings,
              command=command, student_thinking=False, teacher_thinking=False,
              evaluation=dict(gpus=eval_gpus, every=settings['eval_export_freq'] or settings['save_freq'], include_initial=True,
                              n=12, max_tokens=32000, max_model_len=32768))
print(json.dumps(record, indent=2), flush=True)
if sys.argv[2] == '--dry-run':
    raise SystemExit(0)
if not (Path(settings['model_path'])/'config.json').is_file():
    raise SystemExit('Model not bundled. Set MODEL_PATH to a downloaded Qwen/Qwen3-4B directory.')
root.mkdir(parents=True, exist_ok=False)
shutil.copytree(source, snapshot, ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc', 'outputs', 'models', 'data'))
(snapshot/'configs/shell_launch.yaml').write_text(yaml.safe_dump(settings, sort_keys=False))
record['pid'] = os.getpid()
(root/'launch.json').write_text(json.dumps(record, indent=2)+'\n')
reproduce = ('#!/usr/bin/env bash\nset -euo pipefail\ncd -- '+shlex.quote(str(snapshot))+
             '\nexport PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1\nexec '+shlex.join(command)+'\n')
(root/'launch.sh').write_text(reproduce)
print(f'Launching; supervisor log: {root}/supervisor.log', flush=True)
os.chdir(snapshot)
os.environ['PYTHONPATH'] = str(snapshot)
with (root/'supervisor.log').open('a') as log:
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
os.execv(sys.executable, command)
PY

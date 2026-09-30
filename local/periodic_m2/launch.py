"""Portable train/evaluate launcher for the fixed M=2 RL+OPSD baseline."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
TASKS = ('chemistry', 'biology', 'physics', 'materials')


def parse_args(mode, argv=None):
    p = argparse.ArgumentParser(description=f'{mode}: LoRA RL+OPSD with a frozen M=2 teacher')
    p.add_argument('--config', type=Path, default=ROOT / 'configs/rlopsd.yaml')
    p.add_argument('--task', choices=TASKS, required=True)
    p.add_argument('--run-dir', type=Path, help='Default: output_root/<task>-<mode>-<timestamp>')
    p.add_argument('--model-path')
    p.add_argument('--data-root', type=Path)
    p.add_argument('--gpus', help='Physical indices in PCI bus order, e.g. 4,5')
    for flag, kind in [('beta', float), ('steps', int), ('test-freq', int), ('save-freq', int)]:
        p.add_argument('--'+flag, type=kind)
    p.add_argument('--set', action='append', default=[], metavar='KEY=VALUE', help='Override any key in configs/rlopsd.yaml; repeatable')
    p.add_argument('--resume' if mode == 'train' else '--checkpoint', dest='resume', type=Path,
                   help='Complete checkpoints/global_step_N directory; evaluation without it evaluates initial model')
    p.add_argument('--smoke', action='store_true', help='3 steps,4 questions,response128,test/save every1; not a performance run')
    p.add_argument('--render-only', action='store_true', help='Validate and save full configuration without launching GPU workers')
    return p, p.parse_args(argv)


def load_config(p, a):
    defaults = yaml.safe_load((ROOT / 'configs/rlopsd.yaml').read_text())
    supplied = yaml.safe_load(a.config.read_text()) or {}
    if set(supplied)-set(defaults):p.error(f'Unknown config keys: {set(supplied)-set(defaults)}')
    c = {**defaults, **supplied}
    default_profiles = defaults.get('task_overrides', {})
    supplied_profiles = supplied.get('task_overrides', {})
    if not isinstance(supplied_profiles, dict):p.error('task_overrides must be a task-to-settings mapping')
    if set(supplied_profiles)-set(TASKS):p.error('Unknown task in task_overrides')
    profile = {**default_profiles.get(a.task, {}), **supplied_profiles.get(a.task, {})}
    if set(profile)-set(defaults) or 'task_overrides' in profile:p.error('Unknown or nested task override')
    c.pop('task_overrides', None)
    c.update(profile)
    if a.smoke:
        c.update(steps=3, train_batch_size=4, val_batch_size=4, max_response_length=128,
                 val_max_samples=4, test_freq=1, save_freq=1)
    for item in a.set:
        key, sep, value = item.partition('=')
        if not sep or key not in c:p.error(f'Invalid --set: {item}')
        c[key] = yaml.safe_load(value)
    for key in ('model_path', 'data_root', 'gpus', 'beta', 'steps', 'test_freq', 'save_freq'):
        if getattr(a, key) is not None:c[key] = str(getattr(a, key)) if key=='data_root' else getattr(a, key)
    c['data_root'] = str(Path(c['data_root'] or ROOT/'data').expanduser().resolve())
    c['output_root'] = str(Path(c['output_root']).expanduser().resolve())
    c['ray_temp_root'] = str(Path(c['ray_temp_root']).expanduser().resolve())
    if c['cache_dir']:c['cache_dir'] = str(Path(c['cache_dir']).expanduser().resolve())
    for key, value in defaults.items():
        if key == 'task_overrides':continue
        if value is None:continue
        valid = isinstance(c[key], bool) if isinstance(value, bool) else (
            isinstance(c[key], int) and not isinstance(c[key], bool) if isinstance(value, int) else (
            isinstance(c[key], (int, float)) and not isinstance(c[key], bool) if isinstance(value, float) else isinstance(c[key], str)))
        if not valid:p.error(f'{key}: expected {type(value).__name__}, got {c[key]!r}')
    for key in ('steps','train_batch_size','val_batch_size','rollout_n','ppo_mini_batch_size','micro_batch_size',
                'ppo_epochs','max_prompt_length','max_response_length','lora_rank','lora_alpha','tensor_parallel_size',
                'agent_workers','reward_workers','max_num_batched_tokens','max_num_seqs','ray_cpus','threads',
                'total_epochs','log_prob_micro_batch_size','eval_n','max_checkpoints'):
        if c[key] <= 0:p.error(f'{key} must be positive')
    if c['teacher_update_interval'] != 2:p.error('This baseline supports teacher_update_interval=2 only')
    if c['beta'] <= 0 or c['learning_rate'] <= 0:p.error('beta and learning_rate must be positive')
    if not 0 < c['gpu_memory_utilization'] < 1:p.error('gpu_memory_utilization must be between 0 and 1')
    if c['rollout_n'] < 2:p.error('GRPO requires at least 2 rollouts per question')
    gpus = [s.strip() for s in c['gpus'].split(',')]
    if not gpus or not all(s.isdigit() for s in gpus) or len(set(gpus)) != len(gpus):p.error('gpus must be distinct numeric device indices')
    if len(gpus) % c['tensor_parallel_size']:p.error('GPU count must be divisible by tensor_parallel_size')
    if c['train_batch_size'] % c['ppo_mini_batch_size']:p.error('train_batch_size must be divisible by ppo_mini_batch_size')
    if c['ppo_mini_batch_size']*c['rollout_n'] % (len(gpus)*c['micro_batch_size']):p.error('Expanded mini-batch must divide evenly across GPUs and micro-batches')
    if not c['eval_do_sample'] and c['eval_n'] != 1:p.error('Greedy evaluation uses eval_n=1')
    if c['eval_do_sample'] and c['eval_temperature'] <= 0:p.error('Sampled evaluation requires positive eval_temperature')
    return c, gpus


def data_fingerprint(data):
    return {name:hashlib.sha256((data/name).read_bytes()).hexdigest()
            for name in ('train_teacher.parquet','train_plain.parquet','test.parquet')}


def main(mode='train', argv=None):
    p, a = parse_args(mode, argv)
    c, gpus = load_config(p, a)
    root = ROOT
    model = c['model_path']
    data = Path(c['data_root']) / a.task
    for name in ('train_teacher.parquet','train_plain.parquet','test.parquet','manifest.json'):
        if not (data/name).is_file():p.error(f'Missing data: {data/name}')
    if (model.startswith(('/', '.', '~')) and not Path(model).expanduser().is_dir()):p.error(f'Model directory not found: {model}')
    if Path(model).expanduser().is_dir():model = str(Path(model).expanduser().resolve())
    fingerprint = data_fingerprint(data)
    if a.resume:
        a.resume = a.resume.expanduser().resolve()
        if not (a.resume/'actor').is_dir() or not a.resume.name.startswith('global_step_'):p.error('Checkpoint must be a complete global_step_N directory')
        previous = a.resume.parent.parent/'data_fingerprint.json'
        if previous.exists() and json.loads(previous.read_text()) != fingerprint:
            p.error('Checkpoint dataset differs from requested data; do not resume with a changed split')
    if a.run_dir is None:
        stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        a.run_dir = Path(c['output_root']) / f'{a.task}-{mode}-{stamp}'
    a.run_dir = a.run_dir.expanduser().resolve()
    if (a.run_dir/'status.json').exists():p.error('Use a fresh run-dir to preserve previous run records (also for --resume)')
    a.run_dir.mkdir(parents=True, exist_ok=True)

    opts={
     'algorithm.adv_estimator':'grpo', 'algorithm.use_kl_in_reward':False,
     'data.train_files':f'{data}/train_teacher.parquet','data.val_files':f'{data}/test.parquet',
     'data.prompt_key':'prompt','data.seed':c['seed'],'data.dataloader_num_workers':c['data_workers'],
     'data.train_batch_size':c['train_batch_size'],'data.val_batch_size':c['val_batch_size'],
     'data.max_prompt_length':c['max_prompt_length'],'data.max_response_length':c['max_response_length'],
     'data.truncation':'left','data.filter_overlong_prompts':False,
     '+data.apply_chat_template_kwargs.enable_thinking':c['thinking'],
     'data.val_max_samples':c['val_max_samples'],
     'actor_rollout_ref.model.path':model,'actor_rollout_ref.model.lora_rank':c['lora_rank'],
     'actor_rollout_ref.model.lora_alpha':c['lora_alpha'],'actor_rollout_ref.model.target_modules':c['target_modules'],
     'actor_rollout_ref.model.use_remove_padding':True,'actor_rollout_ref.model.enable_gradient_checkpointing':c['gradient_checkpointing'],
     'actor_rollout_ref.actor.strategy':'fsdp','actor_rollout_ref.actor.fsdp_config.model_dtype':'bf16',
     'actor_rollout_ref.actor.fsdp_config.param_offload':c['param_offload'],'actor_rollout_ref.actor.fsdp_config.optimizer_offload':c['optimizer_offload'],
     'actor_rollout_ref.actor.use_torch_compile':False,
     'actor_rollout_ref.actor.use_dynamic_bsz':False,
     'actor_rollout_ref.actor.ppo_mini_batch_size':c['ppo_mini_batch_size'],
     'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu':c['micro_batch_size'],
     'actor_rollout_ref.actor.ppo_epochs':c['ppo_epochs'],
     'actor_rollout_ref.actor.optim.lr':c['learning_rate'],'actor_rollout_ref.actor.optim.lr_warmup_steps':c['warmup_steps'],
     'actor_rollout_ref.actor.optim.weight_decay':c['weight_decay'],
     'actor_rollout_ref.actor.grad_clip':c['grad_clip'],'actor_rollout_ref.actor.entropy_coeff':0,
     'actor_rollout_ref.actor.entropy_checkpointing':c['entropy_checkpointing'],
     'actor_rollout_ref.actor.clip_ratio_low':c['clip_ratio_low'],'actor_rollout_ref.actor.clip_ratio_high':c['clip_ratio_high'],
     'actor_rollout_ref.actor.clip_ratio_c':c['clip_ratio_c'],'actor_rollout_ref.actor.loss_agg_mode':'token-mean',
     'actor_rollout_ref.actor.use_kl_loss':False,
     'actor_rollout_ref.actor.policy_loss.loss_mode':'sdpg',
     'actor_rollout_ref.actor.policy_loss.teacher_update_interval':c['teacher_update_interval'],
     'actor_rollout_ref.actor.policy_loss.alpha':0.0,'actor_rollout_ref.actor.policy_loss.beta':c['beta'],
     'actor_rollout_ref.actor.policy_loss.kl_mode':'urkl',
     'actor_rollout_ref.actor.policy_loss.beta_warmup_steps':0,
     'actor_rollout_ref.actor.policy_loss.beta_decay_steps':0,
     'actor_rollout_ref.actor.policy_loss.beta_distill_positive_advantage_only':False,
     '+actor_rollout_ref.actor.policy_loss.beta_distill_exclude_eos':False,
     'actor_rollout_ref.rollout.load_format':'safetensors',
     'actor_rollout_ref.rollout.layered_summon':True,
     'actor_rollout_ref.rollout.name':'vllm','actor_rollout_ref.rollout.mode':'async',
     'actor_rollout_ref.rollout.agent.num_workers':c['agent_workers'],
     '+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce':True,
     'actor_rollout_ref.rollout.n':c['rollout_n'],'actor_rollout_ref.rollout.temperature':c['temperature'],
     'actor_rollout_ref.rollout.top_p':c['top_p'],'actor_rollout_ref.rollout.top_k':c['top_k'],
     'actor_rollout_ref.rollout.tensor_model_parallel_size':c['tensor_parallel_size'],
     'actor_rollout_ref.rollout.gpu_memory_utilization':c['gpu_memory_utilization'],
     'actor_rollout_ref.rollout.enforce_eager':True,
     'actor_rollout_ref.rollout.free_cache_engine':True,
     'actor_rollout_ref.rollout.max_num_batched_tokens':c['max_num_batched_tokens'],
     'actor_rollout_ref.rollout.max_num_seqs':c['max_num_seqs'],
     'actor_rollout_ref.rollout.max_model_len':c['max_prompt_length']+c['max_response_length'],
     'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu':c['log_prob_micro_batch_size'],
     'actor_rollout_ref.rollout.log_prob_use_dynamic_bsz':False,
     'actor_rollout_ref.rollout.val_kwargs.do_sample':c['eval_do_sample'],
     'actor_rollout_ref.rollout.val_kwargs.temperature':c['eval_temperature'],
     'actor_rollout_ref.rollout.val_kwargs.n':c['eval_n'],
     'reward.custom_reward_function.path':str(root/'local/when_chemistry/reward.py'),
     'reward.custom_reward_function.name':'compute_score',
     'reward.reward_manager.name':'naive','reward.num_workers':c['reward_workers'],
     'trainer.use_legacy_worker_impl':'enable',
     'trainer.n_gpus_per_node':len(gpus),'trainer.nnodes':1,'trainer.total_epochs':c['total_epochs'],
     'trainer.total_training_steps':c['steps'],
     'trainer.val_only':mode=='evaluate',
     'trainer.val_before_train':mode=='evaluate' or (c['val_before_train'] and not bool(a.resume)),'trainer.test_freq':c['test_freq'],
     'trainer.save_freq':-1 if mode=='evaluate' else c['save_freq'],
     'trainer.max_actor_ckpt_to_keep':c['max_checkpoints'],
     'trainer.default_local_dir':str(a.run_dir/'checkpoints'),
     'trainer.validation_data_dir':str(a.run_dir/'evaluation'),
     'trainer.rollout_data_dir':str(a.run_dir/'rollouts'),
     'trainer.logger':'[console,file]',
     'trainer.project_name':'sdpg-periodic-m2','trainer.experiment_name':a.run_dir.name,
     'trainer.resume_mode':'resume_path' if a.resume else 'disable',
     'ray_kwargs.ray_init.num_cpus':c['ray_cpus'],
     '+ray_kwargs.ray_init.object_store_memory':c['ray_object_store_bytes'],
     '+ray_kwargs.ray_init.address':'local',
     '+ray_kwargs.ray_init.include_dashboard':False,
     '+ray_kwargs.ray_init._temp_dir':str(Path(c['ray_temp_root']).expanduser().resolve()/('rlopsd-'+hashlib.sha256(str(a.run_dir).encode()).hexdigest()[:8])),
    }
    if a.resume:opts['trainer.resume_from_path'] = str(a.resume)
    def val(x):
        if isinstance(x, bool):return str(x).lower()
        if isinstance(x, str):return x if x == '[console,file]' else json.dumps(x)
        return str(x)
    cmd = [sys.executable, '-m', 'verl.trainer.main_ppo'] + [f'{k}={val(v)}' for k,v in opts.items()]
    env = os.environ.copy()
    env.update(PYTHONPATH=str(root), CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=','.join(gpus),
               OMP_NUM_THREADS=str(c['threads']), MKL_NUM_THREADS=str(c['threads']), OPENBLAS_NUM_THREADS=str(c['threads']),
               TOKENIZERS_PARALLELISM='false', NCCL_P2P_DISABLE=str(int(c['nccl_p2p_disable'])), NCCL_RAS_ENABLE='0',
               GLOO_SOCKET_IFNAME=c['network_interface'], NCCL_SOCKET_IFNAME=c['network_interface'],
               VLLM_USE_V1='1', VLLM_ATTENTION_BACKEND='FLASH_ATTN', VLLM_USE_DEEP_GEMM='0',
               VLLM_SKIP_FLASHINFER_AUTOTUNE='1', FLASHINFER_DISABLE_VERSION_CHECK='1',
               VERL_FILE_LOGGER_PATH=str(a.run_dir/'metrics.jsonl'), PYTHONUNBUFFERED='1',
               HF_HUB_OFFLINE=str(int(c['offline'])), WANDB_MODE='disabled', PYTHONNOUSERSITE='1')
    env.pop('RAY_ADDRESS', None)
    if c['offline']:
        for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):env.pop(key,None)
    if c['cache_dir']:
        for key, sub in [('HF_HOME','huggingface'),('TRITON_CACHE_DIR','triton'),('TORCHINDUCTOR_CACHE_DIR','inductor')]:
            env[key] = str(Path(c['cache_dir'])/sub)
    record = {'command':shlex.join(cmd), 'cwd':str(root), 'gpu_indices':[int(s) for s in gpus],
              'model':model, 'mode':mode, 'smoke':a.smoke, 'settings':c, 'options':opts,
              'teacher_context_template':'\n\nThe correct answer to this problem is: {answer}\nUse this to verify your reasoning, but show your full solution process.',
              'data_directory':str(data), 'data_fingerprint':fingerprint,
              'evaluation':'greedy n=1 by default; differs from paper mean@16',
              'memory_execution':{'teacher_cpu_offload':'after teacher forward, before log_softmax', 'full_vocabulary_kl_token_chunk':256},
              'environment':{key:env[key] for key in ('CUDA_DEVICE_ORDER','CUDA_VISIBLE_DEVICES','HF_HUB_OFFLINE','OMP_NUM_THREADS','NCCL_P2P_DISABLE','NCCL_SOCKET_IFNAME')}}
    (a.run_dir/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
    (a.run_dir/'settings.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
    (a.run_dir/'data_manifest.json').write_bytes((data/'manifest.json').read_bytes())
    (a.run_dir/'data_fingerprint.json').write_text(json.dumps(fingerprint,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)
    # Always retain the fully resolved Hydra configuration before running.
    result = subprocess.run(cmd+['--cfg','job','--resolve'],cwd=root,env=env,text=True,capture_output=True)
    (a.run_dir/'config-render.log').write_text(result.stdout+'\n'+result.stderr)
    if result.returncode:
        print(result.stderr,file=sys.stderr);return result.returncode
    start = result.stdout.find('model_engine:')
    config_text = result.stdout[start:] if start>=0 else result.stdout
    yaml.safe_load(config_text)
    (a.run_dir/'resolved_config.yaml').write_text(config_text)
    if a.render_only:return 0
    def status(**values):
        (a.run_dir/'status.json').write_text(json.dumps({**values,'time':datetime.datetime.now().isoformat()},indent=2)+'\n')
    status(status='starting',pid=os.getpid(),mode=mode)
    with (a.run_dir/'train.log' if mode=='train' else a.run_dir/'evaluate.log').open('w') as log:
        proc = subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        status(status='process_running',pid=os.getpid(),trainer_pid=proc.pid,mode=mode)
        code = proc.wait()
    status(status='finished' if code==0 else 'failed',returncode=code,mode=mode)
    return code


if __name__ == '__main__':
    sys.exit(main())

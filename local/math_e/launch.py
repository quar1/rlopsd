"""Render the inherited 4B configuration, then launch teacher-first training."""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import shutil

import yaml

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path)
    p.add_argument('--config',type=Path,default=ROOT/'configs/math_e_qwen3_4b.yaml')
    p.add_argument('--method-config',type=Path,default=Path(__file__).with_name('config.yaml'))
    p.add_argument('--prepared-dir',type=Path)
    p.add_argument('--task',choices=['math'],default='math')
    p.add_argument('--model-path')
    p.add_argument('--data-root',type=Path)
    p.add_argument('--gpus',default=os.getenv('TRAIN_GPUS'))
    p.add_argument('--steps',type=int)
    p.add_argument('--save-freq',type=int)
    p.add_argument('--beta',type=float)
    p.add_argument('--set',action='append',default=[],help='Inherited training KEY=VALUE')
    p.add_argument('--method-set',action='append',default=[],help='New method KEY=VALUE')
    p.add_argument('--resume',type=Path)
    p.add_argument('--render-only',action='store_true')
    a=p.parse_args()
    settings=yaml.safe_load(a.config.read_text())
    method=yaml.safe_load(a.method_config.read_text())
    for item in a.method_set:
        key,sep,value=item.partition('=')
        if not sep or key not in method:p.error(f'Unknown method setting {item}')
        method[key]=yaml.safe_load(value)
    if method['probe_loss']!='reference_ce':p.error('Only the existing math reference CE mode is implemented')
    candidates=method['strength_candidates']
    if (not isinstance(candidates,list) or 0. not in candidates or len(set(candidates))!=len(candidates)
            or any(not isinstance(s,(int,float)) or not 0<=s<=1 for s in candidates)
            or method['initial_strength'] not in candidates):p.error('Invalid strength candidates/initial strength')
    method['strength_candidates']=[float(s) for s in candidates]
    method['initial_strength']=float(method['initial_strength'])
    if method['feedback_interval_outer_steps']<=0:p.error('Feedback interval must be positive')
    for key in ('ema_old_teacher_decay','teacher_preference_scale','gaussian_max_strength',
                'gaussian_sigma','cosine_degeneracy_tolerance','feedback_se_multiplier',
                'min_probe_gain','teacher_learning_rate','teacher_weight_decay','teacher_grad_clip'):
        if not isinstance(method[key],(int,float)) or not math.isfinite(method[key]):p.error(f'Nonfinite {key}')
    if (not 0<=method['ema_old_teacher_decay']<=1 or method['teacher_preference_scale']<=0
        or method['gaussian_sigma']<=0 or method['gaussian_max_strength']<0
        or not 0<method['cosine_degeneracy_tolerance']<1 or method['feedback_se_multiplier']<0
        or method['teacher_learning_rate']<=0 or method['teacher_weight_decay']<0 or method['teacher_grad_clip']<=0):
        p.error('Invalid method parameter range')
    for key in ('probe_pool_questions','probe_questions_per_decision','ce_chunk_tokens'):
        if not isinstance(method[key],int) or method[key]<=0:p.error(f'{key} must be a positive integer')
    overrides=list(a.set)
    for env,key in [('MODEL_PATH','model_path'),('DATA_ROOT','data_root'),('BETA','beta'),
                    ('SAVE_EVERY','save_freq'),('EVAL_EVERY','eval_export_freq')]:
        if env in os.environ:overrides.append(key+'='+os.environ[env])
    for key in ('model_path','data_root','save_freq','beta'):
        value=getattr(a,key)
        if value is not None:overrides.append(key+'='+str(value))
    for item in overrides:
        key,_,value=item.partition('=');settings[key]=yaml.safe_load(value)
    a.steps=a.steps if a.steps is not None else int(os.getenv('STEPS',settings['steps']))
    a.gpus=a.gpus or settings['gpus']
    # Retain the published repository's model/data defaults, not local lab paths.
    for key in ('model_path','data_root','output_root','cache_dir','ray_temp_root'):
        if settings.get(key) is None:continue
        path=Path(settings[key]).expanduser()
        settings[key]=str((ROOT/path).resolve() if not path.is_absolute() else path.resolve())
        overrides.append(key+'='+settings[key])
    run=(a.run_dir or Path(settings['output_root'])/('math_e-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f'))).resolve()
    if run==ROOT or ROOT in run.parents:p.error('Use a run directory outside rlopsd to keep source snapshots separate')
    a.prepared_dir=(a.prepared_dir or Path(settings['cache_dir'])/'math_e_v1').expanduser().resolve()
    if not (a.prepared_dir/'split.json').exists():
        from transformers import AutoTokenizer
        from .prepare import prepare
        prepare(Path(settings['data_root'])/'math',a.prepared_dir,
                AutoTokenizer.from_pretrained(settings['model_path'],local_files_only=True),
                method['probe_pool_questions'],method['probe_seed'],
                settings['max_prompt_length']+max(settings['max_response_length'],4096))
    split=json.loads((a.prepared_dir/'split.json').read_text())
    if (split['seed']!=method['probe_seed'] or split['probe_questions']!=min(method['probe_pool_questions'],split['train_questions']//4)
            or split['source']!=str((Path(settings['data_root'])/'math').resolve())):
        p.error('Prepared data differs from request; choose a new --prepared-dir')
    # Render the shared FSDP configuration before adding the math_e worker settings.
    cmd=[sys.executable,'-m','local.math_e.render_config','--config',str(a.config.resolve()),'--task','math',
         '--gpus',a.gpus,'--steps',str(a.steps),'--run-dir',str(run)]
    for item in overrides:cmd+=['--set',item]
    if a.resume:cmd+=['--resume',str(a.resume.resolve())]
    run.mkdir(parents=True,exist_ok=True)
    with (run/'base-render.log').open('w') as log:
        subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    record=json.loads((run/'launch.json').read_text())
    settings=record['settings']
    if (settings['teacher_update_interval']!=-1 or settings['param_offload'] or settings['optimizer_offload']
        or settings['teacher_cpu_offload'] or settings['lr_scheduler_step_unit']!='optimizer'
        or settings['thinking'] or settings['teacher_thinking'] or settings['math_prompt_format']!='native'):
        p.error('math_e requires the inherited nonthinking native-template LoRA configuration without offload')
    if settings['train_batch_size']%len(a.gpus.split(',')):
        p.error('Whole question groups must stay on the same rank')
    from omegaconf import OmegaConf
    config=OmegaConf.load(run/'resolved_config.yaml')
    method.update(prepared_dir=str(a.prepared_dir),output_dir=str(run/'math_e'),rollout_n=settings['rollout_n'])
    config.actor_rollout_ref.actor.policy_loss.math_e=method
    config.actor_rollout_ref.actor.policy_loss._target_='local.math_e.config.MathEPolicyLossConfig'
    config.actor_rollout_ref.rollout.agent.agent_loop_manager_class='local.math_e.rollout.SeededAgentLoopManager'
    config.data.train_files=str(a.prepared_dir/'fit.parquet')
    config.trainer.project_name='math_e'
    # Start with the same seed in model, sampler, and generation workers as the base launch.
    OmegaConf.save(config,run/'resolved_config.yaml')
    (run/'math_e_config.yaml').write_text(yaml.safe_dump(method,sort_keys=False))
    (run/'split.json').write_bytes((a.prepared_dir/'split.json').read_bytes())
    record.update(method=method,teacher_mode=method['method'],
        command=[sys.executable,'-m','local.math_e.entry','--config',str(run/'resolved_config.yaml')],
        evaluation='scripts/train_4b.sh starts asynchronous evaluation; scripts/train.py trains only. AIME24/25/HMMT25 n12 max32000, initial + every20')
    (run/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(run_dir=str(run),method=method,steps=a.steps,gpus=a.gpus),indent=2),flush=True)
    if a.render_only:return 0
    source=run/'source'
    for package in ('verl','local'):
        shutil.copytree(ROOT/package,source/package,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    record['cwd']=str(source)
    (run/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
    env=os.environ.copy();env.update(record['environment'])
    env.update(PYTHONPATH=str(source),PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',
        MKL_NUM_THREADS=str(settings['threads']),OPENBLAS_NUM_THREADS=str(settings['threads']),
        VLLM_USE_V1='1',VLLM_ATTENTION_BACKEND='FLASH_ATTN',VLLM_USE_DEEP_GEMM='0',
        VLLM_SKIP_FLASHINFER_AUTOTUNE='1',FLASHINFER_DISABLE_VERSION_CHECK='1',
        VERL_FILE_LOGGER_PATH=str(run/'metrics.jsonl'),WANDB_MODE='disabled',NCCL_RAS_ENABLE='0',GLOO_SOCKET_IFNAME='lo')
    env.pop('RAY_ADDRESS',None)
    if settings['offline']:
        for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):env.pop(key,None)
    if settings['cache_dir']:
        for key,sub in [('HF_HOME','huggingface'),('TRITON_CACHE_DIR','triton'),('TORCHINDUCTOR_CACHE_DIR','inductor')]:
            env[key]=str(Path(settings['cache_dir'])/sub)
    (run/'status.json').write_text(json.dumps(dict(status='running',pid=os.getpid())))
    with (run/'train.log').open('w') as log:
        result=subprocess.run(record['command'],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
    (run/'status.json').write_text(json.dumps(dict(status='finished' if result.returncode==0 else 'failed',returncode=result.returncode)))
    return result.returncode


if __name__=='__main__':
    sys.exit(main())

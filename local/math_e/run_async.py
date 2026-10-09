"""Fresh math training and persistent evaluation workers, owned sessions only."""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from local.math_e.evaluate import save_json


def members(sid):
    result = []
    assert sid != os.getsid(0)
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        pid = int(path.name)
        try:
            if os.getsid(pid) != sid or path.stat().st_uid != os.getuid():
                continue
            fields = (path/'stat').read_text().rsplit(')', 1)[1].split()
            if fields[0] != 'Z':
                result.append((pid, fields[19]))
        except (FileNotFoundError, ProcessLookupError):
            pass
    return result


def cleanup(sid):
    for sig, timeout in [(signal.SIGTERM, 15), (signal.SIGKILL, 10)]:
        for pid, started in members(sid):
            try:
                fields = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()
                if fields[19] == started and os.getsid(pid) == sid:
                    os.kill(pid, sig)
            except (FileNotFoundError, ProcessLookupError):
                pass
        deadline = time.monotonic()+timeout
        while members(sid) and time.monotonic() < deadline:
            time.sleep(.5)
        if not members(sid):
            return
    raise RuntimeError(f'Owned session {sid} cleanup incomplete')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--config', default='configs/math_e_qwen3_4b.yaml', help='Configuration relative to the immutable run source')
    p.add_argument("--allow-partial-storage-budget", action="store_true", help="Start while old-checkpoint cleanup is pending; stop before disk exhaustion")
    p.add_argument("--eval-gpu", default="2")
    p.add_argument("--eval-gpus", help="Comma-separated independent evaluation GPUs; overrides --eval-gpu")
    p.add_argument("--eval-memory-utilization", type=float, default=.8)
    p.add_argument("--eval-max-num-seqs", type=int, default=8)
    p.add_argument('--wait-for-train-memory', action='store_true', help='Start evaluation now and wait for training memory; never stop existing jobs')
    p.add_argument('--train-min-free-gib', type=float, help='Required free memory per training GPU, including any other job peak reserve')
    p.add_argument('--checkpoint-budget-gib', type=float, default=20)
    a = p.parse_args()
    if a.checkpoint_budget_gib <= 0 or (a.train_min_free_gib is not None and a.train_min_free_gib <= 0):
        p.error('Memory and checkpoint budgets must be positive')
    root = a.run_dir.resolve(); source = root/'source'
    import yaml
    config_path = source/a.config
    settings = yaml.safe_load(config_path.read_text())
    training_gpus = [int(x) for x in settings['gpus'].split(',')]
    eval_gpus = [int(s) for s in (a.eval_gpus or a.eval_gpu).split(',')]
    assert len(set(eval_gpus)) == len(eval_gpus) and not set(eval_gpus) & set(training_gpus)
    assert not (root/'train').exists(), 'Fresh run directory required'
    state = dict(status='preflight', phases={}, supervisor_pid=os.getpid(), started_at=datetime.datetime.now().astimezone().isoformat())
    children = {}; logs = []
    def save():
        state['updated_at'] = datetime.datetime.now().astimezone().isoformat()
        save_json(root/'status.json', state)
    def cancel(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        raise InterruptedError(f'User/supervisor signal {signum}')
    signal.signal(signal.SIGTERM, cancel); signal.signal(signal.SIGINT, cancel)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(source), PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1',
               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', WANDB_MODE='disabled')
    env.pop('RAY_ADDRESS', None)
    def sample():
        value = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.free,utilization.gpu',
                                         '--format=csv,noheader,nounits'], text=True)
        rows = [[int(v.strip()) for v in line.split(',')] for line in value.splitlines()]
        host = next(int(s.split()[1]) for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))/1024**2
        record = dict(time=datetime.datetime.now().astimezone().isoformat(), phase=state.get('phase'),
                      gpu_columns=['index','used_mib','free_mib','util_percent'], gpus=[r for r in rows if r[0] in training_gpus+eval_gpus], host_free_gib=host)
        with (root/'resources.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        return rows, host
    def start(name, args):
        command = [sys.executable]+args
        log = (root/f'{name}.log').open('w'); logs.append(log)
        proc = subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        children[name] = proc
        state['phases'][name] = dict(status='running', pid=proc.pid, session=proc.pid, command=command)
        state.update(status='running', phase=name); save()
        return proc
    def finish(name, proc):
        code = proc.wait(); cleanup(proc.pid)
        state['phases'][name].update(status='complete' if code==0 else 'failed', returncode=code); save()
        children.pop(name)
        if code:
            raise RuntimeError(f'{name} failed with exit {code}; see {root/name}.log')
    model = settings['model_path']
    data = str(Path(settings['data_root'])/'math')
    eval_every = settings.get('eval_export_freq', 0) or settings['save_freq']
    train_args = ['scripts/train.py','--task','math','--config',str(config_path)]
    eval_args = ['-m','local.math_e.evaluate','--model',model,'--data',data,
                 '--prompt-format',settings.get('math_prompt_format','plain'),
                 '--thinking' if settings['thinking'] else '--no-thinking',
                 '--gpu-memory-utilization',str(a.eval_memory_utilization),'--max-num-seqs',str(a.eval_max_num_seqs),'--question-batch-size','4']
    try:
        rows, host = sample(); free = {r[0]:r[2] for r in rows}
        totals = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total','--format=csv,noheader,nounits'], text=True)
        total_mib = {int(row.split(',')[0]):int(row.split(',')[1]) for row in totals.splitlines()}
        training_required_mib = {i: a.train_min_free_gib*1024 if a.train_min_free_gib is not None
                                 else min(65*1024, .9*total_mib[i]) for i in training_gpus}
        if not a.wait_for_train_memory:
            assert all(free[i] > training_required_mib[i] for i in training_gpus), free
        assert all(free[i] > a.eval_memory_utilization*total_mib[i]+2048 for i in eval_gpus), free
        assert host > 80, host
        import shutil
        checkpoint_count = min(settings['max_checkpoints'], math.ceil(settings['steps']/settings['save_freq']))
        required_gib = a.checkpoint_budget_gib*checkpoint_count + 20
        state['storage_budget'] = dict(required_gib=required_gib, free_gib=shutil.disk_usage(root).free/1024**3)
        save()
        minimum_gib = 25 if a.allow_partial_storage_budget or a.wait_for_train_memory else required_gib
        state['storage_budget']['partial_budget_allowed'] = a.allow_partial_storage_budget
        assert shutil.disk_usage(root).free > minimum_gib*1024**3, state['storage_budget']
        def start_evaluators():
            for index, gpu in enumerate(eval_gpus):
                start(f'evaluate_gpu{gpu}', eval_args+['--gpus',str(gpu),'--watch-run',str(root/'train'),
                    '--output',str(root/'evaluation'),'--include-initial','--every',str(eval_every),
                    '--last-step',str(settings['steps']),'--n','12','--max-tokens','32000',
                    '--parallel-worker','--resume-partial','--preferred-step',str(index*eval_every)])
        if a.wait_for_train_memory:
            start_evaluators()
            state['phases']['train'] = dict(status='waiting_resources', required_free_mib=training_required_mib)
            state['phase'] = 'evaluation_running_training_waiting_resources'
            while True:
                rows, host = sample(); free = {r[0]:r[2] for r in rows}
                disk_free_gib = shutil.disk_usage(root).free/1024**3
                state['phases']['train'].update(free_mib={i:free[i] for i in training_gpus},
                    host_free_gib=host, disk_free_gib=disk_free_gib, disk_required_gib=required_gib)
                save()
                for name, proc in list(children.items()):
                    if proc.poll() is not None:
                        finish(name, proc)
                if (all(free[i] > training_required_mib[i] for i in training_gpus)
                        and host > 80 and disk_free_gib > required_gib):
                    break
                time.sleep(10)
        # This entry starts the main training from the configured initial model.
        train = start('train', train_args+['--run-dir',str(root/'train')])
        if not a.wait_for_train_memory:
            start_evaluators()
        state['phase'] = 'formal_train_and_parallel_eval'; save()
        while children:
            sample()
            if 'train' in children and shutil.disk_usage(root).free < 25*1024**3:
                raise RuntimeError('Storage capacity: free space below 25GiB. Preserve checkpoints; free space before resuming. This is not model divergence.')
            for name, proc in list(children.items()):
                if proc.poll() is not None:
                    finish(name, proc)
            time.sleep(5)
        state.update(status='complete', phase='all_done'); save()
    except BaseException as error:
        state.update(status='cancelled' if isinstance(error,InterruptedError) else 'failed', error=type(error).__name__, message=str(error))
        save()
        for name, proc in list(children.items()):
            cleanup(proc.pid)
            state['phases'][name].update(status='stopped', returncode=proc.poll())
        save()
        raise
    finally:
        for stream in logs: stream.close()


if __name__ == '__main__':
    main()

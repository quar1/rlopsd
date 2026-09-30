"""Sequential independent tasks; stop on failure and never carry weights across tasks."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, help='Default: configured output_root/suite-<timestamp>')
    p.add_argument('--tasks', nargs='+', choices=['chemistry','biology','physics','materials'],
                   default=['chemistry','biology','physics','materials'],
                   help='Tasks in execution order (default: chemistry biology physics materials)')
    a, forwarded = p.parse_known_args()
    if any(x.split('=')[0] in ('--resume','--task','--run-dir','--checkpoint') for x in forwarded):
        p.error('Task queue always starts fresh; do not pass task/run-dir/resume/checkpoint')
    if len(set(a.tasks)) != len(a.tasks):p.error('Duplicate tasks')
    if a.root is None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from local.periodic_m2.launch import parse_args, load_config
        cp, ca = parse_args('train', ['--task', a.tasks[0]] + forwarded)
        settings, _ = load_config(cp, ca)
        a.root = Path(settings['output_root']) / ('suite-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    root = a.root.expanduser().resolve()
    if (root/'queue_status.json').exists():p.error('Use a new queue output directory')
    root.mkdir(parents=True,exist_ok=True)
    state = {'order':a.tasks,'status':'starting','tasks':{}}
    def save():
        state['updated_at'] = datetime.datetime.now().isoformat()
        (root/'queue_status.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    for task in a.tasks:
        run = root/task
        cmd = [sys.executable,str(Path(__file__).with_name('launch.py')),'--task',task,'--run-dir',str(run)]+forwarded
        state.update(status='running',task=task)
        state['tasks'][task] = {'command':cmd,'status':'running'}
        save()
        with (root/f'{task}-launcher.log').open('w') as log:
            proc = subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            state['tasks'][task]['pid'] = proc.pid;save()
            code = proc.wait()
        state['tasks'][task].update(status='finished' if code==0 else 'failed',returncode=code)
        if code:
            state.update(status='failed',failure_task=task);save();return code
        save()
    state['status'] = 'configuration_rendered' if '--render-only' in forwarded else 'finished'
    save();return 0


if __name__ == '__main__':
    sys.exit(main())

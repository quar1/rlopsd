"""Math evaluation and persistent checkpoint watcher; labels never guide training."""
import argparse
import json
import os
from pathlib import Path
import time
import fcntl
from contextlib import contextmanager

import pyarrow.parquet as pq
from transformers import AutoTokenizer
from local.math_m2.common import template_kwargs, compute_eval_score
from local.math_m2.official_scoring import EVAL_RULE, TRAIN_RULE


def save_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def summarize(records):
    grouped = {}
    for record in records:
        grouped.setdefault(record['id'], []).append(record)
    assert records and len({len(group) for group in grouped.values()}) == 1
    lengths = sorted(r['length'] for r in records)
    return dict(questions=len(grouped), samples=len(records), n=len(next(iter(grouped.values()))),
                avg_at_n=sum(r['score'] for r in records)/len(records),
                pass_at_n=sum(any(r['score'] for r in group) for group in grouped.values())/len(grouped),
                parse_failure_rate=sum(r['parse_failed'] for r in records)/len(records),
                no_box_rate=sum(r['no_box'] for r in records)/len(records),
                string_match_recovery_rate=sum(r['string_match_recovered'] for r in records)/len(records),
                truncation_rate=sum(r['finish_reason']=='length' for r in records)/len(records),
                mean_length=sum(lengths)/len(lengths), p95_length=lengths[int(.95*(len(lengths)-1))],
                max_length=lengths[-1])


def checkpoint_ready(run, step):
    return evaluation_checkpoint(run, step) is not None


def evaluation_checkpoint(run, step):
    """Resolve a full checkpoint or an atomically published inference-only adapter."""
    latest = run/'checkpoints/latest_checkpointed_iteration.txt'
    try:
        full_published = int(latest.read_text().strip()) >= step
    except (FileNotFoundError, ValueError):
        full_published = False
    root = run/f'checkpoints/global_step_{step}'
    if full_published and all((root/name).is_file() for name in (
        'data.pt', 'actor/periodic_teacher.pt',
        'actor/lora_adapter/adapter_model.safetensors', 'actor/lora_adapter/adapter_config.json')):
        return root
    root = run/f'eval_snapshots/global_step_{step}'
    marker = root/'ready.json'
    if marker.is_file():
        meta = json.loads(marker.read_text())
        if (meta.get('step') == step and meta.get('kind') == 'student_lora_inference_only'
                and all((root/'actor/lora_adapter'/name).is_file()
                        for name in ('adapter_model.safetensors', 'adapter_config.json'))):
            return root
    return None


@contextmanager
def claim_checkpoint(output, step):
    """Process-scoped lock; a crashed worker releases its claim automatically."""
    directory = output/'.claims'
    directory.mkdir(exist_ok=True)
    with (directory/f'step_{step:04d}.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def resume_records(path, rows, n, batch_size):
    """Reuse only complete generation batches; preserve any interrupted tail."""
    if not path.exists():
        return []
    raw = path.read_bytes()
    records, lines = [], []
    for line in raw.splitlines(keepends=True):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            break
        records.append(record)
        lines.append(line)
    group = n*batch_size
    keep = len(records) if len(records) == len(rows)*n else len(records)//group*group
    assert keep <= len(rows)*n
    for index, record in enumerate(records[:keep]):
        assert record['id'] == rows[index//n]['extra_info']['id']
        assert record['sample'] == index % n
    prefix = b''.join(lines[:keep])
    if prefix and not prefix.endswith(b'\n'):
        prefix += b'\n'
    if prefix != raw:
        backup = path.with_name(path.name + f'.interrupted-{time.time_ns()}')
        backup.write_bytes(raw)
        path.write_bytes(prefix)
    return records[:keep]


def evaluate_checkpoint(a, model, tok, output, checkpoint=None, adapter_id=1):
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest
    # Partial results are preserved on failure; never silently overwrite them.
    resuming = output.exists() and getattr(a, 'resume_partial', False)
    output.mkdir(parents=True, exist_ok=resuming)
    request = None
    if checkpoint is not None:
        adapter = checkpoint/'actor/lora_adapter'
        assert (adapter/'adapter_model.safetensors').is_file()
        request = LoRARequest(f'student_{adapter_id}', adapter_id, str(adapter))
    prompt_kwargs = template_kwargs(tok, getattr(a, 'prompt_format', 'plain'), getattr(a, 'thinking', False))
    config = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        'checkpoint': str(checkpoint) if checkpoint else None, 'template': prompt_kwargs['chat_template'],
        'thinking': prompt_kwargs.get('enable_thinking'),
        'temperature': 1., 'top_p': 1., 'top_k': -1, 'min_p': 0., 'presence_penalty': 0.,
        'checkpoint_scope': 'student LoRA inference; never teacher or training resume',
        'reference_reasoning_used': False, 'adapter_id': adapter_id,
        'evaluation_scoring_rule': EVAL_RULE, 'training_scoring_rule': TRAIN_RULE,
    }
    previous_seconds = 0.
    if resuming:
        old = json.loads((output/'config.json').read_text())
        for key in ('model', 'data', 'checkpoint', 'n', 'seed', 'max_tokens', 'max_model_len',
                    'question_batch_size', 'max_questions', 'template', 'temperature', 'top_p',
                    'top_k', 'min_p', 'presence_penalty', 'evaluation_scoring_rule'):
            assert old[key] == config[key], f'Resume configuration mismatch: {key}'
        assert old.get('thinking') == config['thinking'], 'Resume thinking mode differs'
        if (output/'status.json').exists():
            previous_seconds = json.loads((output/'status.json').read_text()).get('seconds', 0.)
        save_json(output/f'resume-{time.time_ns()}.json', config)
    else:
        save_json(output/'config.json', config)
    start = time.monotonic()
    try:
        results = {}
        for task in ('aime24', 'aime25', 'hmmt25'):
            rows = pq.read_table(a.data/'benchmarks'/f'{task}.parquet').to_pylist()
            assert len(rows) == 30
            if a.max_questions:
                rows = rows[:a.max_questions]
            path = output/f'{task}-generations.jsonl'
            records = resume_records(path, rows, a.n, a.question_batch_size) if resuming else []
            with path.open('a' if resuming else 'w') as stream:
                for begin in range(len(records)//a.n, len(rows), a.question_batch_size):
                    batch = rows[begin:begin+a.question_batch_size]
                    prompts, parameters, budgets = [], [], []
                    for offset, row in enumerate(batch):
                        ids = tok.apply_chat_template(row['prompt'], **prompt_kwargs,
                                                     add_generation_prompt=True, tokenize=True)
                        budget = min(a.max_tokens, a.max_model_len-len(ids))
                        assert budget > 0
                        prompts.append({'prompt_token_ids': ids})
                        budgets.append(budget)
                        parameters.append(SamplingParams(n=a.n, temperature=1., top_p=1., top_k=-1,
                            min_p=0., presence_penalty=0., max_tokens=budget, seed=a.seed+begin+offset))
                    outputs = model.generate(prompts, parameters, lora_request=request, use_tqdm=False)
                    assert len(outputs) == len(batch)
                    for row, prompt, budget, generated in zip(batch, prompts, budgets, outputs):
                        assert len(generated.outputs) == a.n
                        for sample, item in enumerate(generated.outputs):
                            record = dict(id=row['extra_info']['id'], sample=sample, text=item.text,
                                length=len(item.token_ids), token_ids=list(item.token_ids),
                                prompt_tokens=len(prompt['prompt_token_ids']), budget=budget,
                                finish_reason=item.finish_reason,
                                **compute_eval_score(task, item.text, row['reward_model']['ground_truth']))
                            stream.write(json.dumps(record, ensure_ascii=False)+'\n')
                            records.append(record)
                    stream.flush()
                    save_json(output/'status.json', dict(status='running', phase=task,
                        completed_questions=min(begin+len(batch),len(rows)),
                        seconds=previous_seconds+time.monotonic()-start))
            results[task] = summarize(records)
            save_json(output/'partial.json', results)
        save_json(output/'results.json', dict(results=results, seconds=previous_seconds+time.monotonic()-start,
            macro_avg_at_n=sum(r['avg_at_n'] for r in results.values())/len(results)))
        save_json(output/'status.json', dict(status='complete', seconds=previous_seconds+time.monotonic()-start))
    except Exception as error:
        save_json(output/'failure.json', dict(error=type(error).__name__, message=str(error)))
        save_json(output/'status.json', dict(status='failed'))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--prompt-format', choices=('plain', 'native'), default='plain')
    p.add_argument('--thinking', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--checkpoint', type=Path)
    mode.add_argument('--watch-run', type=Path, help='Wait for complete training checkpoints and keep one inference engine resident')
    p.add_argument('--every', type=int, default=20)
    p.add_argument('--last-step', type=int, default=200)
    p.add_argument('--include-initial', action='store_true')
    p.add_argument('--parallel-worker', action='store_true')
    p.add_argument('--resume-partial', action='store_true')
    p.add_argument('--preferred-step', type=int)
    p.add_argument('--gpus', required=True)
    p.add_argument('--tensor-parallel-size', type=int, default=1)
    p.add_argument('--n', type=int, default=12)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-tokens', type=int, default=32000)
    p.add_argument('--max-model-len', type=int, default=32768)
    p.add_argument('--gpu-memory-utilization', type=float, default=.6)
    p.add_argument('--max-num-seqs', type=int, default=16)
    p.add_argument('--question-batch-size', type=int, default=4)
    p.add_argument('--max-questions', type=int, default=0, help='Smoke only: limit each benchmark; 0=all30')
    a = p.parse_args()
    assert not a.parallel_worker or a.watch_run
    assert a.n > 0 and a.max_tokens > 0 and a.max_model_len > 0 and a.question_batch_size > 0
    assert a.every > 0 and a.last_step > 0 and a.last_step % a.every == 0
    assert 0 <= a.max_questions <= 30
    devices = a.gpus.split(',')
    assert all(s.isdigit() for s in devices) and len(devices)==a.tensor_parallel_size and len(set(devices))==len(devices)
    os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=a.gpus,
        HF_HUB_OFFLINE='1', VLLM_USE_V1='1', VLLM_USE_DEEP_GEMM='0')
    if a.watch_run:
        a.output.mkdir(parents=True, exist_ok=True)
        status_name = f'worker_gpu{a.gpus}.json' if a.parallel_worker else 'watcher_status.json'
        config_name = f'worker_gpu{a.gpus}_config.json' if a.parallel_worker else 'watcher_config.json'
        save_json(a.output/status_name, dict(status='loading', pid=os.getpid()))
        save_json(a.output/config_name, {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()})
    from vllm import LLM
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    model = LLM(model=a.model, dtype='bfloat16', seed=a.seed, tensor_parallel_size=a.tensor_parallel_size,
        max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_memory_utilization,
        max_num_seqs=a.max_num_seqs, max_num_batched_tokens=4096, enforce_eager=True,
        enable_lora=a.checkpoint is not None or a.watch_run is not None,
        max_lora_rank=64, generation_config='vllm')
    if not a.watch_run:
        evaluate_checkpoint(a, model, tok, a.output, a.checkpoint)
        return
    steps = ([0] if a.include_initial else []) + list(range(a.every, a.last_step+1, a.every))
    if a.parallel_worker:
        try:
            while True:
                pending = [s for s in steps if not (a.output/f'step_{s:04d}/results.json').is_file()]
                if not pending:
                    save_json(a.output/status_name, dict(status='complete', evaluated_steps=steps, pid=os.getpid()))
                    return
                order = sorted(pending, key=lambda s: (s != a.preferred_step, s))
                worked = False
                for step in order:
                    if step and not checkpoint_ready(a.watch_run, step):
                        continue
                    with claim_checkpoint(a.output, step) as claimed:
                        output = a.output/f'step_{step:04d}'
                        if not claimed or (output/'results.json').is_file():
                            continue
                        save_json(a.output/status_name, dict(status='evaluating', step=step, pid=os.getpid()))
                        checkpoint = evaluation_checkpoint(a.watch_run, step) if step else None
                        evaluate_checkpoint(a, model, tok, output, checkpoint, adapter_id=step+1)
                        worked = True
                        break
                if not worked:
                    save_json(a.output/status_name, dict(status='waiting', pending_steps=pending, pid=os.getpid()))
                    training_status = a.watch_run/'status.json'
                    training = json.loads(training_status.read_text()).get('status') if training_status.exists() else None
                    if training in ('failed', 'finished', 'cancelled', 'stopped'):
                        missing = [s for s in pending if s and not checkpoint_ready(a.watch_run, s)]
                        if missing:
                            raise RuntimeError(f'Training {training}; missing checkpoints {missing}')
                    time.sleep(10)
        except Exception as error:
            save_json(a.output/status_name, dict(status='failed', error=type(error).__name__, message=str(error)))
            raise
        return
    try:
        for step in steps:
            output = a.output/f'step_{step:04d}'
            if (output/'results.json').is_file():
                continue
            while step and not checkpoint_ready(a.watch_run, step):
                save_json(a.output/'watcher_status.json', dict(status='waiting_checkpoint', next_step=step, pid=os.getpid()))
                status_path = a.watch_run/'status.json'
                if status_path.exists():
                    try:
                        status = json.loads(status_path.read_text()).get('status')
                    except json.JSONDecodeError:
                        status = None
                    if status in ('failed', 'finished', 'cancelled', 'stopped'):
                        # Recheck readiness: trainer may have finished between our two reads.
                        if not checkpoint_ready(a.watch_run, step):
                            raise RuntimeError(f'Training {status} without complete checkpoint {step}')
                        break
                time.sleep(10)
            save_json(a.output/'watcher_status.json', dict(status='evaluating', step=step, pid=os.getpid()))
            checkpoint = evaluation_checkpoint(a.watch_run, step) if step else None
            evaluate_checkpoint(a, model, tok, output, checkpoint, adapter_id=step+1)
        save_json(a.output/'watcher_status.json', dict(status='complete', evaluated_steps=steps))
    except Exception as error:
        save_json(a.output/'watcher_status.json', dict(status='failed', error=type(error).__name__, message=str(error)))
        raise


if __name__ == '__main__':
    main()

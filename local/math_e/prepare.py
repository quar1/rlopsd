"""Disjoint fit/probe split of the existing training pool; joint target tokenization."""
import argparse
import json
import random
from pathlib import Path

import pyarrow.parquet as pq


def reference_example(tokenizer, row, max_length):
    text = row['prompt'][0]['content']
    problem, sep, context = text.partition('[TEACHER_CONTEXT_TOKEN]')
    if not sep:
        raise ValueError('Math experience mode requires the existing solution-context data')
    start, end = '=== Reference Solution Begin ===\n', '\n=== Reference Solution End ==='
    if start not in context or end not in context:
        raise ValueError('Missing verified dataset solution field in teacher template')
    solution = context.split(start, 1)[1].split(end, 1)[0]
    if not solution.strip():
        raise ValueError('Empty reference solution')
    prefix = tokenizer.apply_chat_template([dict(role='user', content=problem)],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    encoded = tokenizer(prefix + solution, add_special_tokens=False, return_offsets_mapping=True)
    # Match the existing repaired answer-span convention: any overlap belongs to target.
    # No EOS/end tag is appended to the reference target; rollout masks still include EOS.
    mask = [b > len(prefix) and a < len(prefix) + len(solution)
            for a, b in encoded['offset_mapping']]
    if len(mask) > max_length or not any(mask) or mask[0]:
        raise ValueError(f"Invalid/overlong reference: {row['extra_info']['id']}")
    return dict(id=row['extra_info']['id'], input_ids=encoded['input_ids'], target_mask=mask,
                target_tokens=sum(mask), mode='reference_reasoning_ce')


def prepare(source, output, tokenizer, pool_size=128, seed=20261009, max_length=8192):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(source / 'train_teacher.parquet')
    rows = table.to_pylist()
    ids = [r['extra_info']['id'] for r in rows]
    from local.math_e.common import normalize_problem
    keys = [normalize_problem(r['extra_info']['problem']) for r in rows]
    if len(set(ids)) != len(ids) or len(set(keys)) != len(keys):
        raise ValueError('Expected unique questions; never split duplicate questions across fit/probe')
    # On tiny fixtures reserve at most 1/4; real data reserves exactly 128.
    count = min(pool_size, len(rows) // 4)
    if count < 2:
        raise ValueError('Need at least two training probe questions')
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    probe_indices = indices[:count]
    probe_set = set(probe_indices)
    fit_indices = [i for i in range(len(rows)) if i not in probe_set]
    pq.write_table(table.take(fit_indices), output / 'fit.parquet')
    # Prepare all references once: runtime only looks up the current fit batch or rotating probe.
    examples = {r['extra_info']['id']: reference_example(tokenizer, r, max_length) for r in rows}
    import torch
    torch.save(examples, output / 'references.pt')
    split = dict(source=str(source.resolve()), seed=seed, train_questions=len(rows),
        probe_questions=count, probe_fraction=count / len(rows),
        fit_ids=[ids[i] for i in fit_indices], probe_ids=[ids[i] for i in probe_indices],
        probe_order='seeded permutation; rotate without replacement, then wrap',
        target_mode='reference_reasoning_ce', reference_eos_supervised=False,
        reference_boundary='joint tokenization; target-overlapping tokens included',
        provenance='existing prepared solution field; no independent proof of reference correctness')
    (output / 'split.json').write_text(json.dumps(split, ensure_ascii=False, indent=2) + '\n')
    return split


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', required=True, help='Existing downloaded Qwen/Qwen3-4B directory')
    a = p.parse_args()
    from transformers import AutoTokenizer
    split = prepare(a.data, a.output, AutoTokenizer.from_pretrained(a.model, local_files_only=True))
    print(json.dumps({k: v for k, v in split.items() if not k.endswith('_ids')}, indent=2))


if __name__ == '__main__':
    main()

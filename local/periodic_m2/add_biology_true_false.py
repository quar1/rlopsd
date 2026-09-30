"""Extend a fixed 450/50 Biology split with 90/10 true/false examples."""
import argparse
import hashlib
import importlib.util
import json
import random
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def read_jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = args.source
    spec = importlib.util.spec_from_file_location('converter', Path(__file__).parents[1] / 'when_chemistry/prepare_data.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    base = {s: read_jsonl(args.base / f'{s}.jsonl') for s in ('train', 'test')}
    assert len(base['train']) == 450 and len(base['test']) == 50
    added = []
    for row in read_jsonl(source):
        if (row.get('domain'), row.get('details', {}).get('level'), row.get('type')) != ('Biology', 'L3', 'true_or_false'):
            continue
        assert row['answer'] in ('Yes', 'No')
        question = row['question'].strip() + '\nA. Yes\nB. No'
        uid = hashlib.sha256(' '.join(question.split()).encode()).hexdigest()
        added.append({'id': uid, 'question': question,
                      'answer': {'Yes': 'A', 'No': 'B'}[row['answer']],
                      'task': row['details']['task'], 'source_type': 'true_or_false',
                      'original_answer': row['answer']})
    assert len(added) == len({r['id'] for r in added}) == 100
    random.Random(42).shuffle(added)
    additions = {'train': added[10:], 'test': added[:10]}
    splits = {}
    for split in ('train', 'test'):
        splits[split] = base[split] + additions[split]
        random.Random(42).shuffle(splits[split])
    all_ids = [r['id'] for rows in splits.values() for r in rows]
    assert len(all_ids) == len(set(all_ids)) == 600
    assert {r['id'] for r in base['train']} <= {r['id'] for r in splits['train']}
    assert {r['id'] for r in base['test']} <= {r['id'] for r in splits['test']}
    args.output.mkdir(parents=True, exist_ok=False)
    for split, rows in splits.items():
        (args.output / f'{split}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        for privileged in ([True, False] if split == 'train' else [False]):
            records = mod.convert(rows, split, privileged)
            for row, record in zip(rows, records):
                record.update(data_source='sciknoweval_biology', ability='biology')
                record['prompt'][0]['content'] = 'Solve the multiple-choice question. Place your explanation inside <reasoning> tags and the final choice inside <answer> tags. The final choice must be exactly one of the option letters provided in the question.'
                record['extra_info']['source_type'] = row.get('source_type', 'multiple_choice')
            name = ('train_teacher' if privileged else 'train_plain') if split == 'train' else 'test'
            path = args.output / f'{name}.parquet'
            pq.write_table(pa.Table.from_pylist(records), path)
            assert pq.read_table(path).to_pylist() == records
    manifest = {
        'source': str(source), 'source_sha256': mod.sha256(source),
        'base_manifest_sha256': mod.sha256(args.base / 'manifest.json'),
        'base_split_archive': str(args.base.resolve()),
        'seed': 42, 'train': 540, 'test': 60,
        'selection': 'All 600 L3 Biology: 300 four-option + 200 two-option + 100 true/false',
        'split_policy': 'Preserve original 450/50 MCQ membership; shuffle TF in source order with Random(42), first10 test, remaining90 train; shuffle merged splits separately with Random(42)',
        'true_false_mapping': {'A': 'Yes', 'B': 'No'},
        'true_false_answer_field': 'answer (answerKey is empty in source)',
        'added_counts': {s: len(rs) for s, rs in additions.items()},
        'added_label_counts': {s: dict(Counter(r['original_answer'] for r in rs)) for s, rs in additions.items()},
        'exact_author_split_unavailable': True,
        'paper_comparison_note': 'Expanded task: 540/60 differs from When Biology 450/50; do not compare aggregate scores as identical test sets.',
        'teacher_context_policy': 'Answer hint only in train_teacher; train_plain/test prompts contain no answer hint',
        'split_ids': {s: [r['id'] for r in rs] for s, rs in splits.items()},
        'preparer_sha256': mod.sha256(Path(__file__)),
        'files_sha256': {p.name: mod.sha256(p) for p in sorted(args.output.iterdir())},
    }
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'train': 540, 'test': 60,
                      'added_label_counts': manifest['added_label_counts']}, ensure_ascii=False))


if __name__ == '__main__':
    main()

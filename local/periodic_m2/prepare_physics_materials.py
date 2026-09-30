"""Create fixed seed42 L3 Physics/Materials splits in the existing SDPG format."""
import hashlib
import argparse
import importlib.util
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', nargs='+', choices=['physics', 'materials'], default=['physics', 'materials'])
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    source = args.source
    root = args.output_root
    converter = Path(__file__).parents[1] / 'when_chemistry/prepare_data.py'
    spec = importlib.util.spec_from_file_location('chemistry_converter', converter)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # JSON strings may contain Unicode line separators; split only JSONL newlines.
    with source.open() as stream:
        source_rows = [json.loads(line) for line in stream if line.strip()]
    tasks = [t for t in [('physics', 'Physics', 720, 80), ('materials', 'Material', 839, 94)] if t[0] in args.tasks]
    # Do not replace previously fixed splits, including either of these tasks.
    for task, _, _, _ in tasks:
        if (root / task).exists():
            raise FileExistsError(root / task)
    for task, domain, ntrain, ntest in tasks:
        seen = {}
        rows = []
        selected_count = 0
        excluded = []
        duplicates = []
        for r in source_rows:
            if r.get('domain') != domain or r.get('details', {}).get('level') != 'L3':
                continue
            if r.get('type') != 'mcq-4-choices':
                continue
            selected_count += 1
            choices = r['choices']
            if choices['label'] != ['A', 'B', 'C', 'D'] or len(choices['text']) != 4:
                excluded.append({'reason': 'Option text/label mismatch; do not infer a corrected gold label',
                                 'source_record_sha256': hashlib.sha256(json.dumps(r, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                                 'record': r})
                continue
            answer = r['answerKey'].strip().upper()
            assert answer in choices['label']
            question = r['question'].strip() + '\n' + '\n'.join(
                f'{label}. {text}' for label, text in zip(choices['label'], choices['text'])
            )
            assert mod.DELIMITER not in question
            uid = hashlib.sha256(' '.join(question.split()).encode()).hexdigest()
            if uid in seen:
                assert seen[uid] == answer, f'Conflicting answers: {uid}'
                duplicates.append({'id': uid, 'reason': 'Duplicate normalized question/options with identical answer', 'record': r})
                continue
            seen[uid] = answer
            rows.append({'id': uid, 'question': question, 'answer': answer,
                         'task': r.get('details', {}).get('task')})
        assert len(rows) == ntrain + ntest, (task, len(rows))
        random.Random(42).shuffle(rows)
        splits = {'train': rows[ntest:], 'test': rows[:ntest]}
        assert not ({r['id'] for r in splits['train']} & {r['id'] for r in splits['test']})
        out = root / task
        out.mkdir(parents=True, exist_ok=False)
        if excluded:
            (out / 'excluded_invalid.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in excluded))
        if duplicates:
            (out / 'excluded_duplicates.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in duplicates))
        for split, split_rows in splits.items():
            (out / f'{split}.jsonl').write_text(''.join(
                json.dumps(r, ensure_ascii=False) + '\n' for r in split_rows
            ))
            for privileged in ([True, False] if split == 'train' else [False]):
                records = mod.convert(split_rows, split, privileged)
                for record in records:
                    record.update(data_source=f'sciknoweval_{task}', ability=task)
                name = ('train_teacher' if privileged else 'train_plain') if split == 'train' else 'test'
                path = out / f'{name}.parquet'
                pq.write_table(pa.Table.from_pylist(records), path)
                assert pq.read_table(path).to_pylist() == records
        manifest = {
            'source': str(source), 'source_sha256': mod.sha256(source),
            'paper': 'https://arxiv.org/abs/2606.03532',
            'seed': 42, 'train': ntrain, 'test': ntest,
            'selection': f'L3 {domain}, mcq-4-choices only',
            'selected_source_rows': selected_count, 'unique_rows': len(rows),
            'excluded_invalid_count': len(excluded),
            'excluded_duplicate_count': len(duplicates),
            'paper_split_sizes': {'train': 841, 'test': 94} if task == 'materials' else {'train': 720, 'test': 80},
            'split_size_note': 'Materials excludes one malformed 5-text/4-label item and one duplicate; resulting clean split is 839/94 instead of paper 841/94. Test count remains 94.' if excluded else 'Matches paper split sizes; not verified author IDs.',
            'deduplication': 'SHA256 of whitespace-normalized question and options',
            'split_policy': 'Python random.Random(42).shuffle in source order; first ntest rows held out',
            'exact_author_split_unavailable': True,
            'teacher_context_policy': 'Correct-option hint only in train_teacher; absent from train_plain/test prompts',
            'split_ids': {s: [r['id'] for r in rs] for s, rs in splits.items()},
            'preparer_sha256': mod.sha256(Path(__file__)),
            'converter_sha256': mod.sha256(converter),
            'files_sha256': {p.name: mod.sha256(p) for p in sorted(out.iterdir()) if p.is_file()},
        }
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
        print(f'{task}: {ntrain} train / {ntest} test; {out}', flush=True)


if __name__ == '__main__':
    main()

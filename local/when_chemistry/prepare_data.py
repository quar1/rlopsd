"""Convert the existing, fixed Chemistry split to SDPG parquet; never resplit."""
import argparse
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DELIMITER = "[TEACHER_CONTEXT_TOKEN]"
# Keep the system instruction used by our existing Chemistry experiments.
SYSTEM = (
    "Solve the four-option question. Place your explanation inside <reasoning> tags "
    "and the final choice inside <answer> tags. The final choice must be exactly "
    "one of A, B, C, or D."
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def convert(rows, split, privileged):
    result = []
    for index, row in enumerate(rows):
        content = row["question"]
        if privileged:
            content += (
                DELIMITER + "\n\nThe correct answer to this problem is: " + row["answer"]
                + "\nUse this to verify your reasoning, but show your full solution process."
            )
        result.append({
            "data_source": "sciknoweval_chemistry",
            "prompt": [{"role": "system", "content": SYSTEM},
                       {"role": "user", "content": content}],
            "ability": "chemistry",
            "reward_model": {"style": "rule", "ground_truth": row["answer"]},
            "extra_info": {"id": row["id"], "split": split, "index": index},
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    splits = {}
    for split, expected in (("train", 1890), ("test", 210)):
        rows = [json.loads(line) for line in (args.source / f"{split}.jsonl").read_text().splitlines() if line.strip()]
        if len(rows) != expected:
            raise ValueError(f"{split}: expected {expected}, got {len(rows)}")
        for row in rows:
            if row["answer"] not in ("A", "B", "C", "D") or DELIMITER in row["question"]:
                raise ValueError(f"Invalid source row: {row['id']}")
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate IDs in {split}")
        splits[split] = rows
    if {r["id"] for r in splits["train"]} & {r["id"] for r in splits["test"]}:
        raise ValueError("Train/test overlap")
    # Refuse to replace an existing artifact. Re-run with a new output directory.
    args.output.mkdir(parents=True, exist_ok=False)
    artifacts = {}
    for name, split, privileged in (("train_teacher", "train", True),
                                    ("train_plain", "train", False), ("test", "test", False)):
        rows = convert(splits[split], split, privileged)
        path = args.output / f"{name}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        if pq.read_table(path).to_pylist() != rows:
            raise RuntimeError(f"Parquet round-trip failed: {path}")
        artifacts[path.name] = {"rows": len(rows), "sha256": sha256(path), "teacher_context": privileged}
    manifest = {
        "paper": "https://arxiv.org/abs/2606.03532",
        "source_directory": str(args.source.resolve()),
        "source_manifest": json.loads((args.source / "manifest.json").read_text()),
        "source_sha256": {f"{s}.jsonl": sha256(args.source / f"{s}.jsonl") for s in splits},
        "converter_sha256": sha256(Path(__file__)),
        "artifacts": artifacts,
        "split_policy": "Preserve existing seed42 split and row order; author IDs unverified.",
        "test_history": "210-question test has been used in prior project evaluations; not untouched confirmation data.",
        "teacher_context_policy": "Training teacher only; no answer hint in actor or test prompts.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

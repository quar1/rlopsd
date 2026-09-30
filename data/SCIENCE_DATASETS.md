# SciKnowEval L3 fixed splits

Seed42. Chemistry existing split preserved; Biology expanded with true/false while preserving original MCQ membership. Physics/Materials newly prepared on 2026-09-30. These are local fixed splits, not verified original When author IDs.

| Task | Train | Test | Directory |
|---|---:|---:|---|
| chemistry | 1890 | 210 | /data1/wcy/opsd/rlopsd/chemistry |
| biology | 540 | 60 | /data1/wcy/opsd/rlopsd/biology |
| physics | 720 | 80 | /data1/wcy/opsd/rlopsd/physics |
| materials | 839 | 94 | /data1/wcy/opsd/rlopsd/materials |

Each directory has train_teacher.parquet (privileged teacher hint), train_plain.parquet (plain student prompt), test.parquet (no hint), manifest.json. Training variants contain the same questions; they are not separate training sets. Physics/Materials/Biology also contain train.jsonl/test.jsonl. Chemistry JSONL source remains /data1/wcy/opsd/rl/chemistry/.

Materials original935 records: exclude one malformed item with5 option texts but4 labels, and remove one duplicate normalized question/options with identical answer. Clean pool933, local split839/94 (paper841/94). Exclusions preserved as excluded_invalid.jsonl and excluded_duplicates.jsonl; no gold answers guessed or rewritten.

Audit: science_split_audit.json; train/test IDs and normalized question/options do not overlap within each task. Teacher/plain training IDs and labels match; test and plain prompts contain no teacher delimiter.

Biology extension:100 true/false examples map A=Yes, B=No (gold from original answer field).90 added to train,10 to test, seed42. Original450/50 MCQ split archived in /data1/wcy/opsd/rl/biology_mcq500/. Expanded540/60 scores are not the same test as When450/50; extra_info.source_type supports separate reporting. Existing immutable run snapshots/manifests remain unchanged.

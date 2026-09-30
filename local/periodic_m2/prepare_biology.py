"""Create seed42 Biology split once; preserve Chemistry's existing split."""
import argparse, hashlib, importlib.util, json, random
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
source=args.source
out=args.output
spec=importlib.util.spec_from_file_location('prepare',Path(__file__).parents[1]/'when_chemistry/prepare_data.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
seen={};rows=[]
with source.open() as f:
 for line in f:
  r=json.loads(line)
  if r.get('domain','').lower()!='biology' or r.get('details',{}).get('level')!='L3':continue
  c=r.get('choices',{})
  if tuple(c.get('label',[])) not in (('A','B'),('A','B','C','D')) or len(c['text'])!=len(c['label']):continue
  a=r['answerKey'].strip().upper();assert a in ('A','B','C','D')
  options=dict(zip(c['label'],c['text']))
  q=r['question'].strip()+'\n'+'\n'.join(f'{k}. {options[k]}' for k in c['label'])
  uid=hashlib.sha256(' '.join(q.split()).encode()).hexdigest()
  if uid in seen:
   assert seen[uid]==a;continue
  seen[uid]=a;rows.append({'id':uid,'question':q,'answer':a,'task':r.get('details',{}).get('task')})
assert len(rows)==500,len(rows)
random.Random(42).shuffle(rows)
splits={'train':rows[50:],'test':rows[:50]}
out.mkdir(parents=True,exist_ok=False)
for split,rs in splits.items():
 (out/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs))
 for privileged in ([True,False] if split=='train' else [False]):
  records=mod.convert(rs,split,privileged)
  for rec in records:
   rec.update(data_source='sciknoweval_biology',ability='biology')
   rec['prompt'][0]['content']='Solve the multiple-choice question. Place your explanation inside <reasoning> tags and the final choice inside <answer> tags. The final choice must be exactly one of the option letters provided in the question.'
  name=('train_teacher' if privileged else 'train_plain') if split=='train' else 'test'
  pq.write_table(pa.Table.from_pylist(records),out/f'{name}.parquet')
manifest={'source':str(source),'source_sha256':mod.sha256(source),'seed':42,'train':450,'test':50,'exact_author_split_unavailable':True,'selection':'L3 Biology multiple-choice: 300 four-option + 200 two-option; exclude 100 true/false', 'deduplication':'whitespace-normalized question and options','split_ids':{s:[r['id'] for r in rs] for s,rs in splits.items()},'files_sha256':{p.name:mod.sha256(p) for p in out.iterdir() if p.is_file()}}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Biology prepared:',out,'450 train / 50 test')

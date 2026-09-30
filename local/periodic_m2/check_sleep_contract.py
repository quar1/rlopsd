"""Execute production server sleep against a fake engine; no GPU/Ray needed."""
import ast,asyncio
from pathlib import Path
from types import SimpleNamespace
p=Path(__file__).resolve().parents[2]/'verl/workers/rollout/vllm_rollout/vllm_async_server.py'
tree=ast.parse(p.read_text())
cls=next(c for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='vLLMHttpServer')
fn=next(f for f in cls.body if isinstance(f,ast.AsyncFunctionDef) and f.name=='sleep')
ns={'RolloutMode':SimpleNamespace(HYBRID=1,COLOCATED=2,STANDALONE=3)}
exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p),'exec'),ns)
class Engine:
 async def collective_rpc(self,name,kwargs):self.call=(name,kwargs)
async def check():
 for layered,level in [(True,1),(False,2)]:
  e=Engine();x=SimpleNamespace(node_rank=0,config=SimpleNamespace(free_cache_engine=True,layered_summon=layered),rollout_mode=1,engine=e)
  await ns['sleep'](x);assert e.call==('sleep',{'level':level})
asyncio.run(check());print('PASS: server layered_summon preserves immutable base with level1; default remains level2')

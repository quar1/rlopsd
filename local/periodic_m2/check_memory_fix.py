"""Compare production full-vocab outputs/gradients against the failed run's source."""
import argparse
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from verl.workers.actor import dp_actor

torch.manual_seed(42)
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--old-source',type=Path,required=True)
args=parser.parse_args()
old_path=args.old_source
tree=ast.parse(old_path.read_text())
old_node=next(n for cls in tree.body if isinstance(cls,ast.ClassDef) for n in cls.body
              if isinstance(n,ast.FunctionDef) and n.name=='_compute_kl_distill_micro_batch')
ns=dict(dp_actor.__dict__)
exec(compile(ast.Module(body=[old_node],type_ignores=[]),str(old_path),'exec'),ns)
old=ns['_compute_kl_distill_micro_batch']
new=dp_actor.DataParallelPPOActor._compute_kl_distill_micro_batch

class LogitsModel(nn.Module):
    def __init__(self,logits):
        super().__init__();self.logits=nn.Parameter(logits)
    def forward(self,**kwargs):return SimpleNamespace(logits=self.logits*1.0)

B,T,V=2,700,257
base=LogitsModel(torch.randn(B,T+9,V,device='cuda'))
teacher_base=LogitsModel(torch.randn(B,T+13,V,device='cuda')).requires_grad_(False)
responses=torch.randint(V,(B,T),device='cuda')
mask=torch.ones(B,T,device='cuda');mask[1,500:]=0
inputs={'responses':responses,'response_mask':mask,'input_ids':None,'attention_mask':None,'position_ids':None}
results=[]
for fn in [old,new]:
    model=copy.deepcopy(base);teacher=copy.deepcopy(teacher_base)
    wrapper=SimpleNamespace(actor_module=model,periodic_teacher=SimpleNamespace(model=teacher),
        config=SimpleNamespace(entropy_checkpointing=True),device_name='cuda',param_dtype=torch.bfloat16)
    values=fn(wrapper,inputs,inputs,1.0)
    loss=((values[0]+.01*values[1]+.003*values[2])*mask).sum()/mask.sum()
    loss.backward()
    assert teacher.logits.grad is None
    results.append(([x.detach() for x in values],model.logits.grad.detach().clone()))
errors=[float((a-b).abs().max()) for a,b in zip(results[0][0],results[1][0])]
grad_error=float((results[0][1]-results[1][1]).abs().max())
assert max(errors)<1e-6 and grad_error<1e-6,(errors,grad_error)
print(json.dumps({'status':'PASS','seed':42,'tokens':T,'vocab':V,
    'logprob_kl_entropy_max_abs_error':errors,'gradient_max_abs_error':grad_error}))

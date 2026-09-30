"""Real Qwen3-8B 8192-token production KD forward/backward, no checkpoint saved."""
import argparse,json,time
from types import SimpleNamespace
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig,get_peft_model,get_peft_model_state_dict
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.actor.periodic_teacher import PeriodicTeacher

torch.manual_seed(42)
start=time.monotonic()
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model-path',required=True)
parser.add_argument('--response-length',type=int,default=8192)
args=parser.parse_args()
path=args.model_path
lc=LoraConfig(r=64,lora_alpha=128,target_modules='all-linear',task_type='CAUSAL_LM')
def build():
    model=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',local_files_only=True)
    return get_peft_model(model,lc)
student=build().to('cuda')
student.gradient_checkpointing_enable();student.enable_input_require_grads();student.train()
teacher=PeriodicTeacher(build())
teacher.copy_adapter(get_peft_model_state_dict(student))
teacher.begin(1)
opt=torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],lr=1e-5)
T=args.response_length
response=torch.randint(100,2000,(1,T),device='cuda')
def inputs(prompt_len):
    ids=torch.cat([torch.randint(100,2000,(1,prompt_len),device='cuda'),response],dim=1)
    return {'input_ids':ids,'attention_mask':torch.ones_like(ids),
            'position_ids':torch.arange(ids.size(1),device='cuda')[None,:],
            'responses':response,'response_mask':torch.ones_like(response)}
actor=SimpleNamespace(actor_module=student,periodic_teacher=teacher,
    config=SimpleNamespace(entropy_checkpointing=True),device_name='cuda',param_dtype=torch.bfloat16)
torch.cuda.reset_peak_memory_stats()
print('LONG_MEMORY_FORWARD_START',flush=True)
lp,kd,entropy=DataParallelPPOActor._compute_kl_distill_micro_batch(actor,inputs(32),inputs(64),1.0)
assert all(torch.isfinite(x).all() for x in (lp,kd,entropy))
loss=.01*kd.mean()-.001*lp.mean()
loss.backward()
grads=[p.grad for p in student.parameters() if p.grad is not None]
assert grads and all(torch.isfinite(g).all() for g in grads)
opt.step()
teacher.finish(1,lambda:get_peft_model_state_dict(student))
torch.cuda.synchronize()
print(json.dumps({'status':'PASS','model':path,'response_tokens':T,'vocab_size':student.config.vocab_size,
    'loss':float(loss.detach()),'kd':float(kd.mean().detach()),
    'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
    'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
    'seconds':time.monotonic()-start,'note':'Synthetic tokens, unsharded student; not task accuracy or distributed rollout validation'}),flush=True)

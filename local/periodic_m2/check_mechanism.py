"""Production helper/loss regression: six rounds, two mini-updates, odd-step resume."""
import copy, json, tempfile
from pathlib import Path
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from peft import LoraConfig,get_peft_model,get_peft_model_state_dict
from omegaconf import OmegaConf
from verl.workers.actor.periodic_teacher import PeriodicTeacher
from verl.trainer.ppo.core_algos import compute_sdpg_loss
from verl.trainer.ppo.utils import need_reference_policy
from verl.workers.config.actor import ActorConfig,PolicyLossConfig

torch.manual_seed(42)
c=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8)
student=get_peft_model(Qwen3ForCausalLM(c),LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
teacher=PeriodicTeacher(copy.deepcopy(student))
assert not ({p.data_ptr() for p in teacher.model.parameters()} & {p.data_ptr() for p in student.parameters()})
opt=torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],lr=.01)
config=ActorConfig(strategy='fsdp',rollout_n=4,ppo_micro_batch_size_per_gpu=1,policy_loss=PolicyLossConfig(loss_mode='sdpg',alpha=0,beta=.01,teacher_update_interval=2),clip_ratio_high=.28,clip_ratio_c=10)
ids=torch.randint(0,32,(3,6));ids[0,1]=0;mask=torch.tensor([[1,1,1,1,0],[1,1,1,0,0],[1,1,1,1,1.]])
adv=torch.tensor([-1.,0,1.])[:,None].expand(3,5)
seen=[]
with tempfile.TemporaryDirectory() as td:
 for step in range(1,7):
  teacher.begin(step);seen.append(teacher.last_sync_step)
  for mini in range(2):
   opt.zero_grad()
   s=student(ids).logits[:,:-1].float().log_softmax(-1)
   with torch.no_grad():t=teacher.model(ids).logits[:,:-1].float().log_softmax(-1)
   kd=(s.exp()*(s-t)).sum(-1).clamp(max=20);kd.retain_grad()
   lp=s.gather(-1,ids[:,1:,None]).squeeze(-1)
   loss,_=compute_sdpg_loss(torch.full_like(lp,float("nan")),lp,kd,adv,mask,config=config,rollout_log_prob=lp.detach(),responses=ids[:,1:],stop_token_ids=[0])
   loss.backward()
   assert (kd.grad[mask.bool()]>0).all(), 'negative/zero advantages or valid EOS excluded'
   assert (kd.grad[~mask.bool()]==0).all()
   opt.step()
  teacher.finish(step,lambda:get_peft_model_state_dict(student))
  if step==3:
   saved_student=copy.deepcopy(student.state_dict());saved_opt=copy.deepcopy(opt.state_dict())
   teacher.save(Path(td)/'teacher.pt',step)
   fresh=PeriodicTeacher(copy.deepcopy(student))
   fresh.load(Path(td)/'teacher.pt',3)
   assert fresh.last_sync_step==2
   assert teacher.digest(fresh.adapter_state())==teacher.digest(teacher.adapter_state())
   student.load_state_dict(saved_student);opt.load_state_dict(saved_opt);teacher=fresh
assert seen==[0,0,2,2,4,4],seen
cfg=OmegaConf.create({'algorithm':{'use_kl_in_reward':False},'actor_rollout_ref':{'actor':{'use_kl_loss':False,'policy_loss':{'loss_mode':'sdpg','alpha':0}}}})
assert not need_reference_policy(cfg)
# Nonzero alpha must retain reference; upstream non-baseline behavior preserved.
cfg.actor_rollout_ref.actor.policy_loss.alpha=.001;assert need_reference_policy(cfg)
print(json.dumps({'status':'PASS','teacher_used_steps':seen,'sync_steps':[2,4,6],'resume_checkpoint_step':3,'resume_last_sync':2,'negative_zero_kd_gradient':'PASS','padding_gradient_zero':'PASS','reference_disabled_alpha0':'PASS','independent_storage':'PASS'}))

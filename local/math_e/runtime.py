"""Teacher proposals and serial virtual branches around the unchanged SDPG update."""
import copy
import json
import math
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.checkpoint import checkpoint

from .mechanism import StudentSnapshot, choose_strength, cpu_copy, dot, gaussian_mix, norm, rng_state, restore_rng


def global_sum(value, device):
    t = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(t)
    return t.cpu().tolist()


def all_objects(value):
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def finite_metrics(metrics):
    for name, value in metrics.items():
        values = value if isinstance(value, (list, tuple)) else [value]
        for v in values:
            if isinstance(v, (float, int)) and not math.isfinite(v):
                raise FloatingPointError(f'Nonfinite student metric: {name}={v}')


def sequence_nll(model, ids, mask, device, chunk=128):
    """Target-overlap mask on the JOINT tokenization, causal shift, no prompt loss."""
    ids = torch.as_tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    target = torch.as_tensor(mask, dtype=torch.bool, device=device)
    positions = target.nonzero().flatten()
    if not len(positions) or int(positions.min()) < 1:
        raise ValueError('Empty target or no causal context')
    attention = torch.ones_like(ids)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        output = model(input_ids=ids, attention_mask=attention,
            position_ids=torch.arange(ids.shape[1], device=device).unsqueeze(0), use_cache=False)
    logits = output.logits[0, positions - 1].clone()
    labels = ids[0, positions]
    del output
    total = logits.new_zeros((), dtype=torch.float32)
    for start in range(0, len(labels), chunk):
        def ce(x, y):
            return F.cross_entropy(x.float(), y, reduction='sum')
        args = (logits[start:start+chunk], labels[start:start+chunk])
        total = total + (checkpoint(ce, *args, use_reentrant=False) if torch.is_grad_enabled() else ce(*args))
    if not torch.isfinite(total):
        raise FloatingPointError('Nonfinite sequence NLL')
    return total


class ExperienceTeacher:
    """Same interface as PeriodicTeacher, but teacher-first and separately optimized."""
    def __init__(self, model, cfg):
        self.model = model.eval().requires_grad_(False)
        self.cpu_offload = False
        self.interval = -1  # legacy interface only; method schedule belongs to Controller
        self.cfg = cfg
        self.completed_steps = self.last_sync_step = self.optimizer_steps = 0
        self.active_step = None
        self.params = {name.replace('.default.', '.'): p for name, p in model.named_parameters() if 'lora_' in name}
        state = get_peft_model_state_dict(model)
        if set(state) != set(self.params):
            raise ValueError('Unsupported trainable adapter parameterization')
        if any(p.dtype != torch.float32 for p in self.params.values()):
            raise ValueError('Teacher LoRA factors must be FP32')
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
        self.optimizer = torch.optim.AdamW(self.params.values(), lr=cfg['teacher_learning_rate'],
                                          weight_decay=cfg['teacher_weight_decay'])
        warmup = cfg['teacher_warmup_steps']
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
            lambda step: min(step / warmup, 1.) if warmup else 1.)
        self.controller = None

    def adapter_state(self):
        return {k: cpu_copy(p) for k, p in self.params.items()}

    @torch.no_grad()
    def copy_adapter(self, state):
        if set(state) != set(self.params):
            raise ValueError('Teacher/student adapter correspondence differs')
        for key, p in self.params.items():
            if p.shape != state[key].shape or p.dtype != state[key].dtype:
                raise ValueError(f'Adapter shape/dtype mismatch: {key}')
            p.copy_(state[key])

    def state(self):
        return dict(adapter=self.adapter_state(), optimizer=cpu_copy(self.optimizer.state_dict()),
                    scheduler=copy.deepcopy(self.scheduler.state_dict()), optimizer_steps=self.optimizer_steps,
                    buffers={k: cpu_copy(b) for k,b in self.model.named_buffers()},
                    modes={k:m.training for k,m in self.model.named_modules()})

    def restore(self, state):
        self.copy_adapter(state['adapter'])
        self.optimizer.load_state_dict(cpu_copy(state['optimizer']))
        self.scheduler.load_state_dict(copy.deepcopy(state['scheduler']))
        self.optimizer_steps = state['optimizer_steps']
        with torch.no_grad():
            for k,b in self.model.named_buffers(): b.copy_(state['buffers'][k])
        self.model.requires_grad_(False)
        self.optimizer.zero_grad(set_to_none=True)
        for k,m in self.model.named_modules(): m.training = state['modes'][k]

    def begin(self, step):
        if self.active_step is not None or step != self.completed_steps + 1:
            raise RuntimeError('Outer-step mismatch on teacher-first entry')
        self.active_step = step

    def finish(self, step, student_state):
        if self.active_step != step:
            raise RuntimeError('Teacher-first finish without begin')
        if tuple(p._version for p in self.model.parameters()) != self.real_update_versions:
            raise RuntimeError('Teacher mutated during real student optimization')
        if any(p.requires_grad or p.grad is not None for p in self.model.parameters()):
            raise RuntimeError('Teacher must be stop-gradient during student update')
        self.completed_steps = step
        self.active_step = None
        return {'teacher/optimizer_updates_total': self.optimizer_steps}

    def save(self, path, step):
        if self.active_step is not None or step != self.completed_steps:
            raise RuntimeError('Incomplete outer step cannot be checkpointed')
        value = dict(method_version=self.cfg['method_version'], config=self.cfg, teacher=self.state(),
            completed_outer_steps=step, last_sync_step=self.last_sync_step,
            feedback=self.controller.feedback_state(), world_size=dist.get_world_size())
        torch.save(value, str(path)+'.tmp'); os.replace(str(path)+'.tmp', path)

    def load(self, path, expected_step):
        value = torch.load(path, map_location='cpu', weights_only=False)
        if (value['method_version'] != self.cfg['method_version'] or value['completed_outer_steps'] != expected_step
                or value['world_size'] != dist.get_world_size()):
            raise ValueError('Incompatible teacher-first checkpoint')
        for key in self.cfg:
            if key not in ('output_dir',) and value['config'][key] != self.cfg[key]:
                raise ValueError(f'Method config changed on resume: {key}')
        self.restore(value['teacher'])
        self.completed_steps = expected_step
        self.last_sync_step = value['last_sync_step']
        self.active_step = None
        self.controller.restore_feedback(value['feedback'])


class Controller:
    def __init__(self, worker, cfg):
        self.worker, self.actor, self.teacher, self.cfg = worker, worker.actor, worker.periodic_teacher, cfg
        self.student_update = self.actor.update_policy
        self.actor.update_policy = self.update
        self.teacher.controller = self
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.device = torch.device('cuda', torch.cuda.current_device())
        root = Path(cfg['prepared_dir'])
        self.references = torch.load(root/'references.pt', map_location='cpu', weights_only=False)
        self.split = json.loads((root/'split.json').read_text())
        self.probe_order = self.split['probe_ids']
        self.probe_cursor = 0
        self.current_strength = cfg['initial_strength']
        self.window_end = self.student_steps = 0
        self.output = Path(cfg['output_dir'])
        self.output.mkdir(parents=True, exist_ok=True)
        if cfg['probe_questions_per_decision'] % self.world:
            raise ValueError('Probe batch must divide equally across FSDP ranks')

    def feedback_state(self):
        return dict(current_strength=self.current_strength, window_end=self.window_end,
                    probe_cursor=self.probe_cursor, probe_order=self.probe_order, student_optimizer_step=self.student_steps)

    def restore_feedback(self, value):
        if value['probe_order'] != self.probe_order:
            raise ValueError('Probe split/order changed on resume')
        self.current_strength = value['current_strength']; self.window_end = value['window_end']
        self.probe_cursor = value['probe_cursor']; self.student_steps = value['student_optimizer_step']

    @contextmanager
    def measure(self, name, metrics):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        if self.rank==0: print(f'MATH_E_STAGE outer_step={self.teacher.active_step} begin={name}',flush=True)
        start = time.monotonic()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            metrics[name+'_seconds'] = time.monotonic()-start
            metrics[name+'_peak_allocated_gib'] = torch.cuda.max_memory_allocated()/2**30
            if self.rank==0: print(f'MATH_E_STAGE outer_step={self.teacher.active_step} end={name} seconds={metrics[name+"_seconds"]:.2f}',flush=True)

    def groups(self, data, step):
        groups = {}
        rewards = data.batch['token_level_scores'].sum(-1).cpu().tolist()
        for i, (info, score) in enumerate(zip(data.non_tensor_batch['extra_info'], rewards)):
            if score not in (0.,1.):
                raise ValueError(f'Preference requires binary correctness reward, got {score}')
            groups.setdefault(info['id'], []).append((i, score))
        rank_ids = all_objects(list(groups))
        flat_ids = [qid for chunk in rank_ids for qid in chunk]
        if len(flat_ids) != len(set(flat_ids)):
            raise ValueError('A question was split across ranks; disable batch balancing')
        pairs = []
        for qid, entries in groups.items():
            if qid in self.probe_order or len(entries) != self.cfg['rollout_n']:
                raise ValueError('Probe leakage or incomplete rollout group')
            positive = [i for i,r in entries if r == 1]
            negative = [i for i,r in entries if r == 0]
            if positive and negative:
                rng = random.Random(f"{self.cfg['pair_seed']}:{step}:{qid}")
                pairs.append((qid, rng.choice(positive), rng.choice(negative)))
        return groups, pairs

    def trajectory(self, data, i):
        prompt = data.batch['teacher_input_ids'][i][data.batch['teacher_attention_mask'][i].bool()].tolist()
        response = data.batch['responses'][i][data.batch['response_mask'][i].bool()].tolist()
        if not response: raise ValueError('Empty valid rollout')
        return prompt+response, [False]*len(prompt)+[True]*len(response)

    def nll(self, model, example):
        return sequence_nll(model, example['input_ids'], example['target_mask'], self.device,
                            self.cfg['ce_chunk_tokens'])

    def reduced_gradients(self, denominator):
        gradients = []
        for p in self.teacher.params.values():
            g = p.grad.detach().float().clone() if p.grad is not None else torch.zeros_like(p)
            dist.all_reduce(g)
            if denominator: g.div_(denominator)
            gradients.append(g)
        self.teacher.optimizer.zero_grad(set_to_none=True)
        return gradients

    def proposal(self, data, groups, pairs):
        from verl.utils.fsdp_utils import collect_lora_params
        teacher = self.teacher
        old = teacher.state()
        student = cpu_copy(collect_lora_params(self.worker.actor_module_fsdp, layered_summon=False, base_sync_done=True))
        decay = self.cfg['ema_old_teacher_decay']
        # Difference form preserves an exactly identical initial student/teacher.
        ema = {k: v+(1-decay)*(student[k]-v) for k,v in old['adapter'].items()}
        metrics = {}
        try:
            teacher.copy_adapter(ema)
            teacher.model.train()
            for p in teacher.params.values(): p.requires_grad_(True)
            # Qwen3 and this LoRA configuration both have zero dropout. p0 caches are
            # detached scalars evaluated at the EMA start, never aliases of parameters.
            local_pair_loss = 0.
            for _, pos, neg in pairs:
                pos_ids, pos_mask = self.trajectory(data,pos)
                neg_ids, neg_mask = self.trajectory(data,neg)
                with torch.no_grad():
                    p0_pos = -sequence_nll(teacher.model,pos_ids,pos_mask,self.device,self.cfg['ce_chunk_tokens'])
                    p0_neg = -sequence_nll(teacher.model,neg_ids,neg_mask,self.device,self.cfg['ce_chunk_tokens'])
                pos_score = -sequence_nll(teacher.model,pos_ids,pos_mask,self.device,self.cfg['ce_chunk_tokens'])
                neg_score = -sequence_nll(teacher.model,neg_ids,neg_mask,self.device,self.cfg['ce_chunk_tokens'])
                margin = (pos_score-p0_pos)-(neg_score-p0_neg)
                loss = F.softplus(-self.cfg['teacher_preference_scale']*margin)
                loss.backward(); local_pair_loss += float(loss.detach())
                del loss, pos_score, neg_score
            pair_count = int(global_sum(len(pairs),self.device))
            experience = self.reduced_gradients(pair_count)
            local_ref_loss = 0.
            for qid in groups:
                example = self.references[qid]
                loss = self.nll(teacher.model,example)/example['target_tokens']
                loss.backward(); local_ref_loss += float(loss.detach())
                del loss
            question_count = int(global_sum(len(groups),self.device))
            reference = self.reduced_gradients(question_count)
            mixed, geometry = gaussian_mix(experience,reference,self.cfg['gaussian_max_strength'],
                self.cfg['gaussian_sigma'],self.cfg['cosine_degeneracy_tolerance'])
            metrics.update(geometry)
            metrics.update(preference_loss=global_sum(local_pair_loss,self.device)/max(pair_count,1),
                reference_loss=global_sum(local_ref_loss,self.device)/question_count,
                pair_count=pair_count, no_pair_questions=question_count-pair_count, reference_coverage=1.)
            used_lr = teacher.optimizer.param_groups[0]['lr']
            if not geometry['skip_optimizer']:
                for p,g in zip(teacher.params.values(),mixed): p.grad=g
                metrics['teacher_grad_norm'] = float(torch.nn.utils.clip_grad_norm_(
                    list(teacher.params.values()),self.cfg['teacher_grad_clip'],error_if_nonfinite=True))
                teacher.optimizer.step(); teacher.scheduler.step(); teacher.optimizer_steps += 1
            teacher.optimizer.zero_grad(set_to_none=True)
            teacher.model.eval().requires_grad_(False)
            candidate = teacher.state()
            keys = list(teacher.params)
            delta = [candidate['adapter'][k]-old['adapter'][k] for k in keys]
            metrics.update(teacher_lr=used_lr, ema_displacement=norm([ema[k]-old['adapter'][k] for k in keys]),
                gradient_candidate_displacement=norm([candidate['adapter'][k]-ema[k] for k in keys]),
                proposal_displacement=norm(delta),
                experience_dot_proposal=dot([g.cpu() for g in experience],delta),
                reference_dot_proposal=dot([g.cpu() for g in reference],delta))
            if not all(torch.isfinite(v).all() for v in candidate['adapter'].values()):
                raise FloatingPointError('Nonfinite teacher candidate parameters')
            return dict(old=old,candidate=candidate,metrics=metrics,
                        experience_grad=cpu_copy(experience),reference_grad=cpu_copy(reference))
        finally:
            teacher.restore(old)

    def apply(self, proposal, strength):
        if strength == 0:
            if proposal: self.teacher.restore(proposal['old'])
            return
        if proposal is None: raise RuntimeError('Nonzero strength without current proposal')
        self.teacher.restore(proposal['candidate'])
        self.teacher.copy_adapter({k: v+strength*(proposal['candidate']['adapter'][k]-v)
                                   for k,v in proposal['old']['adapter'].items()})

    def displacement(self, snapshot):
        return math.sqrt(global_sum(snapshot.squared_displacement(),self.device))

    def probe_ids(self):
        count = min(self.cfg['probe_questions_per_decision'],len(self.probe_order))
        # Match FSDP forward counts across ranks, without padding/duplicating evidence.
        count -= count % self.world
        ids = [self.probe_order[(self.probe_cursor+i)%len(self.probe_order)] for i in range(count)]
        self.probe_cursor = (self.probe_cursor+count)%len(self.probe_order)
        return ids

    @torch.no_grad()
    def probe(self, ids):
        model = self.actor.actor_module
        modes = [(m,m.training) for m in model.modules()]
        model.eval()
        try:
            local = []
            for qid in ids[self.rank::self.world]:
                row = self.references[qid]
                local.append((qid,float(self.nll(model,row)/row['target_tokens'])))
            result = dict(x for part in all_objects(local) for x in part)
            return [result[qid] for qid in ids]
        finally:
            for m,mode in modes: m.training=mode

    def update(self, data):
        step = int(data.meta_info['global_steps'])
        decision = (step-1)%self.cfg['feedback_interval_outer_steps']==0
        if not decision and step > self.window_end:
            raise RuntimeError('Feedback window missing on resume')
        metrics = dict(outer_step=step, decision_step=decision)
        groups,pairs = self.groups(data,step)
        stats = global_sum([len(groups),len(pairs),sum(sum(v for _,v in rows)==0 for rows in groups.values()),
            sum(sum(v for _,v in rows)==len(rows) for rows in groups.values())],self.device)
        metrics.update(question_count=stats[0],pair_count=stats[1],no_pair_questions=stats[0]-stats[1],
                       all_wrong_fraction=stats[2]/stats[0],all_correct_fraction=stats[3]/stats[0],
                       mixed_fraction=stats[1]/stats[0])
        lengths=[];repeats=[];boxed=[]
        from local.math_e.common import last_boxed
        for ids,mask in zip(data.batch['responses'],data.batch['response_mask']):
            tokens=ids[mask.bool()].cpu().tolist();lengths.append(len(tokens))
            grams=[tuple(tokens[i:i+3]) for i in range(max(0,len(tokens)-2))]
            repeats.append(1-len(set(grams))/len(grams) if grams else 0.)
            boxed.append(last_boxed(self.worker.tokenizer.decode(tokens,skip_special_tokens=True)) is not None)
        rollout_stats=global_sum([len(lengths),sum(lengths),sum(x>=data.batch['responses'].shape[1] for x in lengths),
            sum(repeats),sum(boxed),float(data.batch['token_level_scores'].sum())],self.device)
        metrics.update(response_tokens_mean=rollout_stats[1]/rollout_stats[0],
            reached_length_limit_rate=rollout_stats[2]/rollout_stats[0],
            repeated_token_trigram_fraction=rollout_stats[3]/rollout_stats[0],
            complete_box_format_rate=rollout_stats[4]/rollout_stats[0],reward_mean=rollout_stats[5]/rollout_stats[0])
        # Capture BEFORE any teacher/probe computation: every student branch starts
        # from this exact optimizer/RNG state. No driver/sampler/rollout calls here.
        snapshot = StudentSnapshot(self.actor)
        proposal = None
        if decision or self.current_strength > 0:
            with self.measure('teacher_proposal',metrics):
                proposal=self.proposal(data,groups,pairs)
            metrics.update(proposal['metrics'])
        else:
            metrics.update(proposal_skipped=True,preference_loss=None,reference_loss=None,reference_coverage=None)
        decision_record = None
        if decision:
            ids=self.probe_ids()
            losses,displacements,branch_details={},{},{}
            for strength in self.cfg['strength_candidates']:
                snapshot.restore(); self.apply(proposal,strength)
                branch={}
                try:
                    with self.measure('virtual_update',branch):
                        virtual_metrics=self.student_update(data)
                    finite_metrics(virtual_metrics)
                    displacements[strength]=self.displacement(snapshot)
                    with self.measure('probe_forward',branch):
                        losses[strength]=self.probe(ids)
                    branch.update(student_optimizer_updates=virtual_metrics['actor/optimizer_updates'],
                                  student_displacement=displacements[strength])
                    branch_details[str(strength)]=branch
                finally:
                    snapshot.restore(); self.teacher.restore(proposal['old'])
            self.current_strength,decision_record=choose_strength(losses,displacements,self.current_strength,
                self.cfg['feedback_se_multiplier'],self.cfg['min_probe_gain'])
            self.window_end=step+self.cfg['feedback_interval_outer_steps']-1
            tokens=int(global_sum(int(data.batch['response_mask'].sum()),self.device))
            decision_record.update(outer_step=step,probe_ids=ids,probe_cursor_after=self.probe_cursor,
                branches=branch_details,virtual_completion_tokens=tokens*self.actor.config.ppo_epochs*len(losses),
                probe_target_tokens=sum(self.references[i]['target_tokens'] for i in ids)*len(losses),
                probe_mode=self.cfg['probe_loss'])
        snapshot.restore()
        self.apply(proposal,self.current_strength)
        self.teacher.model.eval().requires_grad_(False)
        self.teacher.real_update_versions=tuple(p._version for p in self.teacher.model.parameters())
        committed=bool(proposal and self.current_strength>0 and not proposal['metrics']['skip_optimizer'])
        if self.current_strength>0: self.teacher.last_sync_step=step
        with self.measure('real_student_update',metrics):
            real_metrics=self.student_update(data)
        finite_metrics(real_metrics)
        self.student_steps+=int(real_metrics['actor/optimizer_updates'])
        metrics.update(student_optimizer_step=self.student_steps,teacher_optimizer_step=self.teacher.optimizer_steps,
            teacher_optimizer_committed=committed,strength=self.current_strength,
            effective_ema_old_teacher_decay=1-self.current_strength*(1-self.cfg['ema_old_teacher_decay']),
            window_start=self.window_end-self.cfg['feedback_interval_outer_steps']+1,window_end=self.window_end,
            student_displacement=self.displacement(snapshot))
        if proposal:
            old=proposal['old']['adapter']; now=self.teacher.adapter_state()
            actual=[now[k]-old[k] for k in old]
            metrics.update(teacher_actual_displacement=norm(actual),
                experience_dot_actual=dot(proposal['experience_grad'],actual),
                reference_dot_actual=dot(proposal['reference_grad'],actual))
        else:
            metrics.update(teacher_actual_displacement=0.,experience_dot_actual=0.,reference_dot_actual=0.)
        if decision_record:
            expected=decision_record['branches'][str(self.current_strength)]['student_optimizer_updates']
            if real_metrics['actor/optimizer_updates'] != expected:
                raise RuntimeError('Real and virtual optimizer update counts differ')
        # Preserve losses and full per-branch diagnostics outside the scalar metric logger.
        metrics['student_metrics']=real_metrics
        if self.rank==0:
            if decision_record:
                (self.output/f'feedback_{step:06d}.json').write_text(json.dumps(decision_record,indent=2,allow_nan=False)+'\n')
            with (self.output/'teacher_first.jsonl').open('a') as stream:
                stream.write(json.dumps(metrics,allow_nan=False)+'\n')
            print('MATH_E '+json.dumps({k:v for k,v in metrics.items() if k!='student_metrics'},allow_nan=False),flush=True)
        real_metrics.update({'math_e/'+k:float(v) for k,v in metrics.items() if isinstance(v,(int,float))})
        return real_metrics

"""Pure geometry, paired selection, and complete mutable student rollback."""
import copy
import math
import random

import numpy as np
import torch


def cpu_copy(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: cpu_copy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [cpu_copy(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(cpu_copy(v) for v in obj)
    return copy.deepcopy(obj)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(s):
    random.setstate(s['python'])
    np.random.set_state(s['numpy'])
    torch.set_rng_state(s['cpu'].cpu())
    if s['cuda']:
        torch.cuda.set_rng_state_all([v.cpu() for v in s['cuda']])


def dot(a, b):
    # Accumulate over ALL tensors; scalar statistics in FP64.
    return float(sum((x.double() * y.double()).sum() for x, y in zip(a, b)))


def norm(a):
    return math.sqrt(max(0., dot(a, a)))


def trajectory_seeds(indices,step,seed):
    counts={};seeds=[]
    for index in indices:
        index=int(index);sample=counts.get(index,0);counts[index]=sample+1
        seeds.append((int(seed)+step*1_000_000_007+index*1009+sample)%(2**63-1))
    return seeds


def gaussian_mix(experience, reference, gamma_max=.3, sigma=.5, tolerance=1e-8):
    if not (sigma > 0 and gamma_max >= 0 and 0 < tolerance < 1):
        raise ValueError('Invalid Gaussian parameters')
    if len(experience) != len(reference) or not experience:
        raise ValueError('Gradient structures differ/are empty')
    for e, r in zip(experience, reference):
        if e.shape != r.shape or not torch.isfinite(e).all() or not torch.isfinite(r).all():
            raise FloatingPointError('Invalid teacher gradients')
    e = [g.float() for g in experience]
    r = [g.float() for g in reference]
    ne, nr = norm(e), norm(r)
    info = dict(experience_grad_norm=ne, reference_grad_norm=nr, cosine=0., gamma=0.,
                conflict=False, fallback='none', skip_optimizer=False)
    if ne == 0 and nr == 0:
        out = [torch.zeros_like(g) for g in e]
        info.update(fallback='both_zero', skip_optimizer=True)
    elif nr == 0:
        out = e
        info['fallback'] = 'experience_only'
    elif ne <= max(1e-20, 1e-6 * nr):
        out = r
        info['fallback'] = 'reference_fallback'
    else:
        c = min(1., max(-1., dot(e, r) / (ne * nr)))
        u, v = [g / ne for g in e], [g / nr for g in r]
        residual = [b - c * a for a, b in zip(u, v)] if c >= 0 else [a - c * b for a, b in zip(u, v)]
        residual_norm = norm(residual)
        info.update(cosine=c, conflict=c < 0, residual_norm=residual_norm)
        if c >= 1 - tolerance or (c >= 0 and residual_norm <= math.sqrt(tolerance)):
            out = e
            info['fallback'] = 'same_direction'
        elif c <= -1 + tolerance or residual_norm <= math.sqrt(tolerance):
            out = [torch.zeros_like(g) for g in e]
            info.update(fallback='opposite_direction', skip_optimizer=True)
        else:
            gamma = gamma_max * math.exp(-c * c / (2 * sigma * sigma))
            if c >= 0:
                out = [ne * (a + gamma * q / residual_norm) for a, q in zip(u, residual)]
            else:
                gamma = min(gamma, math.sqrt(max(0., 1 - c * c)) / (-c))
                out = [ne * (b / residual_norm + gamma * z) for b, z in zip(residual, v)]
            info['gamma'] = gamma
    info.update(experience_dot_direction=dot(e, out), reference_dot_direction=dot(r, out))
    if not all(torch.isfinite(g).all() for g in out):
        raise FloatingPointError('Nonfinite mixed gradient')
    return out, info


def choose_strength(losses, displacements, previous, se_multiplier=1., min_gain=0.):
    """losses: strength -> matched, ordered per-question CE values."""
    if 0. not in losses:
        raise ValueError('Missing frozen-teacher virtual-learning baseline')
    base = np.asarray(losses[0.], dtype=np.float64)
    if any(not np.isfinite(v).all() for v in map(np.asarray, losses.values())):
        raise FloatingPointError('Nonfinite probe loss')
    if any(not math.isfinite(v) for v in displacements.values()):
        raise FloatingPointError('Nonfinite virtual displacement')
    if any(len(v) != len(base) for v in losses.values()):
        raise ValueError('Unpaired probe results')
    reason = 'insufficient_probe_questions' if len(base) < 2 else (
        'all_virtual_displacements_zero' if all(v == 0 for v in displacements.values()) else None)
    scores = {}
    for strength, values in losses.items():
        delta = base - np.asarray(values)
        gain = float(delta.mean()) if len(delta) else 0.
        se = float(delta.std(ddof=1) / math.sqrt(len(delta))) if len(delta) > 1 else 0.
        scores[str(strength)] = dict(losses=list(values), paired_delta=delta.tolist(),
                                    gain=gain, se=se, score=gain - se_multiplier * se)
    chosen = previous if reason else 0.
    if not reason:
        eligible = [s for s in losses if s > 0 and scores[str(s)]['score'] > min_gain]
        if eligible:
            chosen = min(eligible, key=lambda s: (-scores[str(s)]['score'], s))
    return chosen, dict(candidates=scores, selected=chosen, feedback_uninformative=bool(reason), reason=reason)


class StudentSnapshot:
    """Local FSDP optimizer shards, buffers, optimizer, scheduler, scaler, RNG and modes.

    Frozen base tensors are immutable; storing them again would not add state isolation.
    FSDP must be between complete forward/backward calls when capturing/restoring.
    """
    def __init__(self, actor):
        self.actor = actor
        self.params = [(p, cpu_copy(p), cpu_copy(p.grad)) for p in actor.actor_module.parameters() if p.requires_grad]
        self.buffers = [(b, cpu_copy(b)) for b in actor.actor_module.buffers()]
        self.modes = [(m, m.training) for m in actor.actor_module.modules()]
        self.optimizer = cpu_copy(actor.actor_optimizer.state_dict())
        self.scheduler = cpu_copy(actor.optimizer_lr_scheduler.state_dict())
        self.scaler = cpu_copy(actor.scaler.state_dict()) if actor.scaler else None
        self.rng = rng_state()
        self.attributes = {name: (hasattr(actor, name), cpu_copy(getattr(actor, name, None)))
                           for name in ('gradient_accumulation', 'optimizer_lrs_used')}
        self.global_batch_info = copy.deepcopy(actor.config.global_batch_info)

    @torch.no_grad()
    def restore(self):
        for p, value, grad in self.params:
            if p.shape != value.shape:
                raise RuntimeError('FSDP shard shape changed during virtual update')
            p.copy_(value)
            p.grad = None if grad is None else grad.to(p.device).clone()
        for b, value in self.buffers:
            b.copy_(value)
        self.actor.actor_optimizer.load_state_dict(cpu_copy(self.optimizer))
        self.actor.optimizer_lr_scheduler.load_state_dict(copy.deepcopy(self.scheduler))
        if self.scaler is not None:
            self.actor.scaler.load_state_dict(copy.deepcopy(self.scaler))
        for m, training in self.modes:
            m.training = training
        for name, (exists, value) in self.attributes.items():
            if exists:
                setattr(self.actor, name, copy.deepcopy(value))
            elif hasattr(self.actor, name):
                delattr(self.actor, name)
        self.actor.config.global_batch_info.clear()
        self.actor.config.global_batch_info.update(self.global_batch_info)
        restore_rng(self.rng)

    def squared_displacement(self):
        return sum(float((p.detach().double().cpu() - v.double()).square().sum()) for p, v, _ in self.params)

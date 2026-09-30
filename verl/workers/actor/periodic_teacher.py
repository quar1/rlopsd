"""Independent frozen LoRA teacher, synchronized only after whole rollout updates."""
import hashlib
import json
import os
import time

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict


class PeriodicTeacher:
    def __init__(self, model, interval=2):
        if interval != 2:
            raise ValueError("This baseline is restricted to M=2")
        self.model = model.eval().requires_grad_(False)
        self.interval = interval
        self.completed_steps = 0
        self.last_sync_step = 0
        self.active_step = None

    def adapter_state(self):
        return {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(self.model).items()}

    @staticmethod
    def digest(state):
        h = hashlib.sha256()
        for key, value in sorted(state.items()):
            h.update(key.encode())
            h.update(value.contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    def copy_adapter(self, state):
        expected = get_peft_model_state_dict(self.model)
        if set(state) != set(expected):
            raise ValueError(f"Teacher/student adapter keys differ: {set(state) ^ set(expected)}")
        set_peft_model_state_dict(self.model, state)
        self.model.eval().requires_grad_(False)
        for key, value in get_peft_model_state_dict(self.model).items():
            if not torch.equal(value.detach().cpu(), state[key].detach().cpu().to(value.dtype)):
                raise RuntimeError(f"Teacher copy failed: {key}")

    def begin(self, step):
        if self.active_step is not None or step != self.completed_steps + 1:
            raise RuntimeError(f"Teacher iteration mismatch: requested {step}, completed {self.completed_steps}")
        self.active_step = step
        self.before_hash = self.digest(self.adapter_state())
        self.before_versions = tuple(p._version for p in self.model.parameters())
        self.model.eval()

    def finish(self, step, student_state):
        if self.active_step != step:
            raise RuntimeError("finish must follow begin for this outer iteration")
        if tuple(p._version for p in self.model.parameters()) != self.before_versions:
            raise RuntimeError("Teacher parameters were mutated inside student optimization")
        if self.digest(self.adapter_state()) != self.before_hash:
            raise RuntimeError("Teacher adapter changed inside student optimization")
        if any(p.grad is not None or p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("Teacher must have no trainable parameters or gradients")
        start = time.monotonic()
        synced = step % self.interval == 0
        used_teacher_step = self.last_sync_step
        if synced:
            self.copy_adapter(student_state())
            self.last_sync_step = step
        self.completed_steps = step
        self.active_step = None
        event = {"event": "teacher_sync" if synced else "teacher_frozen", "completed_outer_step": step,
                 "teacher_used_from_step": used_teacher_step, "last_sync_step": self.last_sync_step,
                 "before_sha256": self.before_hash, "after_sha256": self.digest(self.adapter_state()),
                 "sync_seconds": time.monotonic() - start}
        print("PERIODIC_TEACHER " + json.dumps(event), flush=True)
        return {"teacher/synced": float(synced), "teacher/used_from_step": used_teacher_step,
                "teacher/last_sync_step": self.last_sync_step, "teacher/sync_seconds": event["sync_seconds"]}

    def save(self, path, step):
        if self.active_step is not None or step != self.completed_steps:
            raise RuntimeError("Checkpoint requested before outer update completed")
        state = {"teacher_adapter": self.adapter_state(), "completed_outer_steps": self.completed_steps,
                 "last_sync_step": self.last_sync_step, "teacher_update_interval": self.interval}
        tmp = str(path) + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, path)

    def load(self, path, expected_step):
        state = torch.load(path, map_location="cpu", weights_only=True)
        step = state["completed_outer_steps"]
        if (step != expected_step or state["teacher_update_interval"] != self.interval
                or state["last_sync_step"] != step - step % self.interval):
            raise RuntimeError("Checkpoint teacher schedule mismatch")
        self.copy_adapter(state["teacher_adapter"])
        self.completed_steps = step
        self.last_sync_step = state["last_sync_step"]
        self.active_step = None

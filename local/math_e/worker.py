"""Teacher-first worker reusing the framework's FSDP and rollout lifecycle."""
from omegaconf import OmegaConf
from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
from .runtime import Controller, ExperienceTeacher


class MathEWorker(AsyncActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if not self._is_actor:
            return
        if (self._is_offload_param or self._is_offload_optimizer or not self._is_lora
                or self.config.actor.strategy != 'fsdp' or self.actor.scaler is not None
                or not hasattr(self.actor,'optimizer_lr_scheduler')):
            raise ValueError('math_e requires LoRA FSDP, BF16/FP32, optimizer-step scheduler, no offload')
        cfg=OmegaConf.to_container(self.config.actor.policy_loss.math_e,resolve=True)
        model=self.periodic_teacher.model
        model.to(self.actor.actor_optimizer.param_groups[0]['params'][0].device)
        self.periodic_teacher=ExperienceTeacher(model,cfg)
        self.actor.periodic_teacher=self.periodic_teacher
        self.math_e_controller=Controller(self,cfg)
        print('MATH_E_WORKER_READY rank='+str(self.rank),flush=True)

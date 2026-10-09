"""Use the project's original RayPPOTrainer and generation/checkpoint managers."""
import argparse
from pathlib import Path
import ray
from omegaconf import OmegaConf
from verl.trainer.main_ppo import TaskRunner, run_ppo


class MathETaskRunner(TaskRunner):
    def add_actor_rollout_worker(self, config):
        from local.math_e.worker import MathEWorker
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role
        self.role_worker_mapping[Role.ActorRollout]=ray.remote(MathEWorker)
        self.mapping[Role.ActorRollout]='global_pool'
        return MathEWorker,RayWorkerGroup


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    a=p.parse_args()
    run_ppo(OmegaConf.load(a.config),task_runner_class=ray.remote(num_cpus=1)(MathETaskRunner))


if __name__=='__main__':
    main()

"""Per-trajectory sampling seeds survive restart without a mutable engine RNG cursor."""
import ray
import numpy as np
import json
import time
from pathlib import Path
from .mechanism import trajectory_seeds
from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AgentLoopWorker


class SeededAgentLoopWorker(AgentLoopWorker):
    async def _run_agent_loop(self, sampling_params, trajectory, **kwargs):
        request_seed=int(kwargs['math_e_sampling_seed'])
        return await super()._run_agent_loop({**sampling_params,'seed':request_seed},trajectory,**kwargs)


class SeededAgentLoopManager(AgentLoopManager):
    def __init__(self,*args,**kwargs):
        self.agent_loop_workers_class=ray.remote(SeededAgentLoopWorker)
        super().__init__(*args,**kwargs)

    def generate_sequences(self,prompts):
        # Assign ordinals BEFORE splitting across workers; a question can straddle chunks.
        indices=prompts.non_tensor_batch['index']
        step=int(prompts.meta_info['global_steps'])
        seeds=trajectory_seeds(indices,step,self.config.data.seed)
        prompts.non_tensor_batch['math_e_sampling_seed']=np.asarray(seeds,dtype=object)
        from .memory import rollout_memory
        root=Path(self.config.actor_rollout_ref.actor.policy_loss.math_e.output_dir)
        kwargs=dict(directory=str(root),step=step)
        ray.get([s.collective_rpc.remote(rollout_memory,kwargs=dict(kwargs,reset=True)) for s in self.server_handles])
        start=time.monotonic()
        result=super().generate_sequences(prompts)
        ray.get([s.collective_rpc.remote(rollout_memory,kwargs=dict(kwargs,reset=False)) for s in self.server_handles])
        memory=[json.loads(p.read_text()) for p in sorted(root.glob(f'rollout_memory_{step:06d}_*.json'))]
        record=dict(outer_step=step,rollout_seconds=time.monotonic()-start,memory_by_process=memory,
                    sampling_seeds=seeds,question_indices=[int(i) for i in indices])
        (root/f'rollout_{step:06d}.json').write_text(json.dumps(record,indent=2)+'\n')
        return result

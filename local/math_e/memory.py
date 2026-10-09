"""Executed inside each vLLM tensor-parallel process through collective_rpc."""


def rollout_memory(worker, reset, directory, step):
    import json
    import os
    from pathlib import Path
    import torch
    torch.cuda.synchronize()
    if reset:
        torch.cuda.reset_peak_memory_stats()
        return
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    value=dict(outer_step=step,pid=os.getpid(),cuda_device=torch.cuda.current_device(),
               peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
               peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
               scope='vLLM TP process PyTorch allocator; includes resident model/KV cache')
    (root/f'rollout_memory_{step:06d}_{os.getpid()}.json').write_text(json.dumps(value)+'\n')

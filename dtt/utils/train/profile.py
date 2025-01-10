import torch.profiler

import os


class NullProfiler:
    """A do-nothing profiler that matches the interface of torch.profiler."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def step(self):
        pass


def get_profiler(rank, enabled):
    if enabled:
        os.makedirs("profile", exist_ok=True)

        schedule = torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1)

        trace_handler = torch.profiler.tensorboard_trace_handler(
            "profile", f"rank{rank}", use_gzip=True
        )

        return torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=schedule,
            on_trace_ready=trace_handler,
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
            with_flops=True,
        )

    else:
        return NullProfiler()

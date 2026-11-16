"""Opt-in synchronized wall-clock profiling; separate from normal latency runs."""

from collections import defaultdict
from contextlib import contextmanager
from time import perf_counter

import torch


class StageProfiler:
    """Non-overlapping CPU+GPU stages, including transfers and host computation.

    Synchronizing each boundary perturbs execution. These are diagnostic times,
    not a substitute for uninstrumented end-to-end matcher latency.
    """

    def __init__(self, device):
        self.device = torch.device(device)
        self.events = []
        self.synchronize()
        self.started = perf_counter()

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    @contextmanager
    def section(self, stage, depth):
        self.synchronize()
        start = perf_counter()
        try:
            yield
        finally:
            self.synchronize()
            self.events.append(
                {
                    "stage": stage,
                    "depth": depth,
                    "milliseconds": (perf_counter() - start) * 1000,
                }
            )

    def finish(self):
        self.synchronize()
        total = (perf_counter() - self.started) * 1000
        stages = defaultdict(float)
        for event in self.events:
            stages[event["stage"]] += event["milliseconds"]
        stages["other"] = max(0.0, total - sum(stages.values()))
        return {"total_ms": total, "stages_ms": dict(stages), "events": self.events}

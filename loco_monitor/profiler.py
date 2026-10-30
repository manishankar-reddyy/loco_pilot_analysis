"""Per-stage timing profiler."""
from __future__ import annotations

import time
import torch
from collections import defaultdict


class PipelineProfiler:
    def reset(self):
        self.__init__()

    def __init__(self):
        self.timings = defaultdict(float)
        self.counts = defaultdict(int)
        self.events = []
        self.stage_stack = []

    def cuda_sync(self):
        try:
            if 'torch' in globals() and torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

    def start(self, name, sync_cuda=False):
        if sync_cuda:
            self.cuda_sync()
        self.stage_stack.append((name, time.time(), sync_cuda))

    def stop(self, extra=None):
        name, t0, sync_cuda = self.stage_stack.pop()
        if sync_cuda:
            self.cuda_sync()
        dt = time.time() - t0
        self.timings[name] += dt
        self.counts[name] += 1
        item = {'stage': name, 'seconds': round(dt, 4)}
        if extra:
            item.update(extra)
        self.events.append(item)
        return dt

    def add(self, name, seconds, extra=None):
        self.timings[name] += float(seconds)
        self.counts[name] += 1
        item = {'stage': name, 'seconds': round(float(seconds), 4)}
        if extra:
            item.update(extra)
        self.events.append(item)

    def snapshot_gpu(self):
        try:
            if 'torch' in globals() and torch.cuda.is_available():
                return {
                    'gpu_name': torch.cuda.get_device_name(0),
                    'vram_allocated_gb': round(torch.cuda.memory_allocated()/1e9, 3),
                    'vram_reserved_gb': round(torch.cuda.memory_reserved()/1e9, 3),
                    'vram_total_gb': round(torch.cuda.get_device_properties(0).total_memory/1e9, 3),
                }
        except Exception:
            pass
        return {}

    def report(self):
        rows = []
        for k, v in sorted(self.timings.items(), key=lambda kv: -kv[1]):
            rows.append({'stage': k, 'seconds': round(v, 3), 'calls': self.counts[k]})
        return {'timings': rows, 'events': self.events, 'gpu': self.snapshot_gpu()}

PROFILER = PipelineProfiler()

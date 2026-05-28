import os
import time
import json
import torch
from pathlib import Path
from contextlib import contextmanager
from transformers import TrainerCallback
from datetime import datetime

# --- Global Singleton Storage ---
_CURRENT_PROFILER = None


class BenchmarkLogger:
    def __init__(self, run_name: str, phase: str, output_dir: str = "data/performance_profiling"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = self.output_dir / \
            f"{timestamp}_{run_name}_{phase}.jsonl"
        self.enabled = True

        # Buffer for writing
        print(f"[Benchmark] Logging to {self.filepath}")

    def log(self, event: str, duration_s: float, metadata: dict = None):
        if not self.enabled:
            return

        entry = {
            "timestamp": time.time(),
            "event": event,
            "duration_s": round(duration_s, 4),
            "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            **(metadata or {})
        }

        # Reset peak memory stats so we get fresh peaks for the next block
        torch.cuda.reset_peak_memory_stats()

        with open(self.filepath, "a") as f:
            f.write(json.dumps(entry) + "\n")


def setup_benchmarking(run_name: str, phase: str, is_main_process: bool = True):
    """Initializes the global profiler. Call this once at the start of a phase."""
    global _CURRENT_PROFILER
    if is_main_process:
        _CURRENT_PROFILER = BenchmarkLogger(run_name, phase)
    else:
        # distinct "dummy" or None for non-main processes to avoid file write conflicts
        _CURRENT_PROFILER = None


@contextmanager
def benchmark_scope(name: str, metadata: dict = None):
    """Context manager to time a block of code."""
    if _CURRENT_PROFILER is None:
        yield
        return

    start = time.perf_counter()
    yield
    duration = time.perf_counter() - start
    _CURRENT_PROFILER.log(name, duration, metadata)


class BenchmarkingCallback(TrainerCallback):
    """Automatically logs step time and throughput for Trainer."""

    def __init__(self, packing_seq_len: int = None):
        self.step_start = None
        self.packing_seq_len = packing_seq_len

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if _CURRENT_PROFILER and self.step_start:
            duration = time.perf_counter() - self.step_start

            # Calculate tokens per second (Approximate)
            # Batch Size * Grad Accum * World Size * Seq Len
            total_batch_size = args.per_device_train_batch_size * \
                args.gradient_accumulation_steps * args.world_size
            tokens_per_sec = (total_batch_size * self.packing_seq_len) / \
                duration if self.packing_seq_len else 0

            _CURRENT_PROFILER.log("train_step", duration, {
                "step": state.global_step,
                "tokens_per_sec": round(tokens_per_sec, 2),
                "samples_per_sec": round(total_batch_size / duration, 2)
            })

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if _CURRENT_PROFILER and metrics:
            _CURRENT_PROFILER.log("validation_loop", metrics.get("eval_runtime", 0), {
                "eval_samples_per_second": metrics.get("eval_samples_per_second"),
                "eval_loss": metrics.get("eval_loss")
            })

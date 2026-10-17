"""B3 helper: a few seconds of sustained FP32 matmul load on GPU 0 (no tracker, no files written).
Usage: rl-exp/bin/python contexts/step9/scratch/b3_load.py [seconds]   (default 20)"""
import sys, time, torch

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
a = torch.randn(8192, 8192, device="cuda")
b = torch.randn(8192, 8192, device="cuda")
torch.cuda.synchronize()
t0, n = time.perf_counter(), 0
while time.perf_counter() - t0 < secs:
    for _ in range(10):
        a @ b
    torch.cuda.synchronize()
    n += 10
print(f"load done: {n} matmuls (8192x8192 fp32) in {time.perf_counter() - t0:.1f} s")

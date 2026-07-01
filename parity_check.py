#!/usr/bin/env python3
"""
parity_check.py — the notorch CUDA correctness gate (run on a GPU pod).

The real proof that the CUDA path is correct is NOT "gpu_dispatch_count > 0" (any cuBLAS
call bumps that). It is: the SAME model + SAME input produces the SAME loss / logits / grad
on the CPU path and the GPU path. One CUDA build exposes both via nt_set_gpu_mode(0/1), so we
toggle in-process — identical weights (same seed), only the compute substrate differs. Any
divergence = a GPU/CPU coherence bug (stale mirror, missing sync, or a wrong kernel).

Requires a CUDA build: run with NANO_REQUIRE_CUDA=1 so a silent BLAS fallback aborts.
"""
import os, sys, math
NANO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, NANO); os.chdir(NANO)
os.environ.setdefault('NANO_REQUIRE_CUDA', '1')
import ariannamethod.notorch_nn as nn
from ariannamethod.notorch_nn import NotorchNanoAGI, NotorchEngine, seed as nt_seed, _lib

print(f"backend={nn.NOTORCH_BACKEND} gpu={nn.NOTORCH_GPU}")
if not nn.NOTORCH_GPU:
    print("PARITY: FAIL — GPU not active (need a CUDA build on a GPU box)"); sys.exit(1)

V, ctx = 512, 32
ARCH = dict(n_embd=64, n_head=4, n_layer=2, ctx=ctx, n_content=2, n_rrpram=2)
x = [t % V for t in range(ctx)]
y = [t % V for t in range(1, ctx + 1)]


def run(gpu):
    _lib.nt_set_gpu_mode(1 if gpu else 0)
    nt_seed(7)                                   # identical xavier init in both modes
    m = NotorchNanoAGI(V, **ARCH)
    eng = NotorchEngine(m, lr=3e-4)
    loss = eng.step(x, y, update=False)          # forward-only loss
    logits = eng.forward_logits(x)               # full forward, last-row logits
    ga, gn = eng.grad_check(x, y, 1, 5, eps=1e-3)  # analytic grad on a non-embed param
    return loss, logits, ga


loss_c, logits_c, ga_c = run(gpu=False)          # CPU path (inside the CUDA build)
loss_g, logits_g, ga_g = run(gpu=True)           # GPU path
_lib.nt_set_gpu_mode(1)

dloss = abs(loss_c - loss_g)
dlogit = max(abs(a - b) for a, b in zip(logits_c, logits_g))
dgrad = abs(ga_c - ga_g)
print(f"loss   cpu={loss_c:.6f} gpu={loss_g:.6f}  |Δ|={dloss:.6e}")
print(f"logits max|Δ|={dlogit:.6e}  (cpu[0]={logits_c[0]:.4f} gpu[0]={logits_g[0]:.4f})")
print(f"grad   cpu={ga_c:.6f} gpu={ga_g:.6f}  |Δ|={dgrad:.6e}")
print(f"gpu_dispatch_count={nn.gpu_dispatch_count()}")

ok = dloss < 1e-3 and dlogit < 1e-2 and dgrad < 1e-3 and nn.gpu_dispatch_count() > 0
print("PARITY:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)

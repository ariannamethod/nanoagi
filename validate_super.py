#!/usr/bin/env python3
"""
validate_super.py — point-C validation of the supermilestone artifact (CPU/BLAS, neo).

The A40 trained super.nt and reported it. This proves the SAVED FILE is a correct,
loadable, trained-flesh checkpoint on a *different machine and backend* — the real
"weights survived the pod" test, KARL-independent (the run's KARL was not persisted;
that gap is fixed forward in train_supermilestone.py, but the WEIGHTS stand on their
own and are what ship).

Four checks, all machine-verified:
  1. manifest ↔ file: arch fields present, load path's fail-closed guard accepts it.
  2. load-integrity: NotorchEngine.load(super.nt) returns True; every tensor size matched.
  3. trained-flesh: forward_logits on fixed ids is finite AND mean|logit| ≈ the pod's
     final |flesh| (~2.26), NOT the ~0.43 of an untrained init. Random init can't fake this.
  4. fail-closed R5: loading against a WRONG arch (n_layer+1) is REJECTED (returns None),
     not silently mis-loaded.

Run: NANO_CKPT=pod_run/super.nt python3 validate_super.py
"""
import os, sys, math
NANO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, NANO); os.chdir(NANO)
import nanoagi as N
import ariannamethod.notorch_nn as nn
from ariannamethod.notorch_nn import NotorchNanoAGI, NotorchEngine, seed as nt_seed

CKPT = os.environ.get('NANO_CKPT', 'pod_run/super.nt')
man = N._read_manifest(CKPT)
assert man is not None, f"no manifest for {CKPT}"
print(f"backend={nn.NOTORCH_BACKEND} gpu={nn.NOTORCH_GPU}")
print(f"manifest: {man}")

A = dict(n_embd=man['n_embd'], n_head=man['n_head'], n_layer=man['n_layer'],
         ctx=man['ctx'], n_content=man['n_content'], n_rrpram=man['n_rrpram'])
V = man['vocab_size']
fails = []

# ── build a model at the manifest arch; confirm the load-path guard accepts it ──
nt_seed(42)
m = NotorchNanoAGI(V, **A)
eng = NotorchEngine(m)
dedup = sum(p.numel for p in eng.params)
print(f"built model: dedup params={dedup:,}  (manifest n_params={man['n_params']:,})")

# a Python-shaped shim so _arch_matches / _get_notorch_engine can be exercised as prod does
class _PyShim:
    vocab_size=V; n_embd=A['n_embd']; n_head=A['n_head']; n_layer=A['n_layer']
    context_len=A['ctx']; n_content=A['n_content']; n_rrpram=A['n_rrpram']
shim = _PyShim()
if not N._arch_matches(man, shim):
    fails.append("check1: _arch_matches rejected the manifest at its own arch")
else:
    print("check1 PASS: manifest arch self-consistent, fail-closed guard accepts it")

# ── load-integrity ──
if not eng.load(CKPT):
    fails.append("check2: eng.load(super.nt) returned False (size mismatch / unreadable)")
else:
    print("check2 PASS: load-integrity — every tensor size matched, weights loaded")

# ── trained-flesh magnitude (the pod ended at |flesh|=2.2579; untrained init ~0.43) ──
import random
random.seed(999)
ctx = A['ctx']
tot = 0.0; n = 16; bad = False
for _ in range(n):
    ids = [random.randint(0, V - 1) for _ in range(ctx)]
    row = eng.forward_logits(ids)
    if any(math.isnan(x) or math.isinf(x) for x in row):
        bad = True; break
    tot += sum(abs(x) for x in row) / len(row)
fm = tot / n
if bad:
    fails.append("check3: forward_logits produced NaN/Inf")
elif fm < 1.0:
    fails.append(f"check3: |flesh|={fm:.4f} looks untrained (<1.0; init is ~0.43)")
else:
    print(f"check3 PASS: trained-flesh — forward finite, mean|logit|={fm:.4f} "
          f"(pod final 2.2579; untrained init ~0.43)")

# ── fail-closed R5: wrong arch must be REJECTED ──
class _WrongShim(_PyShim):
    n_layer = A['n_layer'] + 1
if N._arch_matches(man, _WrongShim()):
    fails.append("check4: _arch_matches ACCEPTED a wrong arch (n_layer+1) — guard is open")
else:
    print("check4 PASS: fail-closed R5 — wrong arch (n_layer+1) rejected")

# ── integrity: file sha256 + size match the manifest (catches corruption/truncation) ──
if 'sha256' not in man or 'bytes' not in man:
    fails.append("check5: manifest lacks sha256/bytes integrity fields")
else:
    import hashlib
    h = hashlib.sha256()
    with open(CKPT, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    digest, size = h.hexdigest(), os.path.getsize(CKPT)
    if digest != man['sha256'] or size != man['bytes']:
        fails.append(f"check5: integrity mismatch — sha/size differ from manifest")
    else:
        print(f"check5 PASS: integrity — sha256 {digest[:16]}… + {size} bytes match manifest")

# ── KARL sidecar: the weights are tokenizer-specific, so a shippable checkpoint must ship
#    its KARL. load_state must reconstruct the SAME vocab the weights were trained against. ──
sidecar = CKPT + '.karl'
if not os.path.exists(sidecar):
    fails.append(f"check6: KARL sidecar {sidecar} missing — weights have no tokenizer (not shippable)")
else:
    k = N.KARL()
    if not k.load_state(sidecar):
        fails.append("check6: KARL sidecar failed to load")
    elif k.vocab_size != V:
        fails.append(f"check6: KARL vocab {k.vocab_size} != manifest vocab {V}")
    else:
        print(f"check6 PASS: KARL sidecar loads, vocab={k.vocab_size} matches manifest")

print("\n" + ("VALIDATE: FAIL — " + "; ".join(fails) if fails else "VALIDATE: PASS (6/6)"))
sys.exit(1 if fails else 0)

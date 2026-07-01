#!/usr/bin/env python3
"""
train_supermilestone.py — the nanoagi notorch supermilestone trainer.

One organism, one optimizer, one runtime: KARL (growing BPE) tokenizes, Chuck
(notorch C) trains a 10-15M-param dual-attention transformer, and the trained
flesh generates. No PyTorch anywhere — pure notorch (CPU / BLAS / CUDA, auto-
detected). The full KARL <-> Chuck pipeline, end to end.

The SAME script runs a tiny smoke on neo and the full run on an A40 pod — every
knob is an env var with a pod-scale default. Everything machine-verified lands in
the paper-log (NANO_PAPERLOG), molequla-style: the numbers, not the narrative.

Env (defaults = full pod run):
  NANO_EMBD=384 NANO_HEAD=6 NANO_LAYER=5 NANO_CTX=256 NANO_CONTENT=3 NANO_RRPRAM=3
  NANO_MERGES=2048            # KARL max_merges (frozen during training)
  NANO_STEPS=12000            # training steps (Karpathy-scaled for ~13M params)
  NANO_LR=3e-4
  NANO_VAL_EVERY=500          # val + sample + crossover-metric interval
  NANO_CORPUS_MB=150          # download this many MB of climbmix if no file given
  NANO_CORPUS_FILE=           # pre-staged corpus (skips download); else ~/arianna-datasets/
  NANO_CKPT=karl_chuck_supermilestone.nt
  NANO_PAPERLOG=nanoagi_supermilestone_paperlog.md
"""
import os, sys, time, math, random, resource

NANO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, NANO)
os.chdir(NANO)
import nanoagi as N
import ariannamethod.notorch_nn as nn
from ariannamethod.notorch_nn import NotorchNanoAGI, NotorchEngine, seed as nt_seed


def _env_int(k, d):   return int(os.environ.get(k, d))
def _env_float(k, d): return float(os.environ.get(k, d))


CFG = dict(
    embd=_env_int('NANO_EMBD', 384), head=_env_int('NANO_HEAD', 6),
    layer=_env_int('NANO_LAYER', 5), ctx=_env_int('NANO_CTX', 256),
    content=_env_int('NANO_CONTENT', 3), rrpram=_env_int('NANO_RRPRAM', 3),
    merges=_env_int('NANO_MERGES', 2048), steps=_env_int('NANO_STEPS', 12000),
    karl_mb=_env_int('NANO_KARL_MB', 5),   # KARL learns merges on this many MB (they generalize)
    lr=_env_float('NANO_LR', 3e-4), val_every=_env_int('NANO_VAL_EVERY', 500),
    corpus_mb=_env_int('NANO_CORPUS_MB', 150),
    corpus_file=os.environ.get('NANO_CORPUS_FILE', ''),
    ckpt=os.environ.get('NANO_CKPT', 'karl_chuck_supermilestone.nt'),
    paperlog=os.environ.get('NANO_PAPERLOG', 'nanoagi_supermilestone_paperlog.md'),
)

_PLOG = open(CFG['paperlog'], 'a')


def plog(msg):
    print(msg, flush=True)
    _PLOG.write(msg + "\n")
    _PLOG.flush()


def rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == 'darwin' else r / 1e3   # mac:bytes, linux:KB


# ── 1. Corpus ────────────────────────────────────────────────────────────────
def build_corpus():
    if CFG['corpus_file'] and os.path.exists(CFG['corpus_file']):
        data = open(CFG['corpus_file'], 'rb').read()
        plog(f"corpus: {CFG['corpus_file']} ({len(data)/1e6:.1f} MB, pre-staged)")
        return data
    if CFG['corpus_mb'] <= 0:
        data = open(N.KARL_TXT, 'rb').read()
        plog(f"corpus: karl.txt ({len(data)/1e6:.2f} MB, smoke)")
        return data
    # download climbmix to target size
    target = CFG['corpus_mb'] * 1_000_000
    ds_dir = os.path.expanduser('~/arianna-datasets')
    os.makedirs(ds_dir, exist_ok=True)
    out_path = os.path.join(ds_dir, f"climbmix_{CFG['corpus_mb']}mb.txt")
    if os.path.exists(out_path) and os.path.getsize(out_path) >= target * 0.9:
        data = open(out_path, 'rb').read()
        plog(f"corpus: {out_path} ({len(data)/1e6:.1f} MB, cached)")
        return data
    plog(f"corpus: downloading {CFG['corpus_mb']} MB climbmix -> {out_path}")
    chunks, got, t0 = [], 0, time.time()
    while got < target:
        batch = N._download_climbmix_batch(num_docs=100)
        if not batch:
            plog(f"  download stalled at {got/1e6:.1f} MB; using what we have")
            break
        for t in batch:
            b = (t + "\n\n").encode('utf-8', 'replace')
            chunks.append(b); got += len(b)
        if len(chunks) % 1000 == 0:
            plog(f"  {got/1e6:.1f}/{CFG['corpus_mb']} MB  [{time.time()-t0:.0f}s]")
    data = b''.join(chunks)
    with open(out_path, 'wb') as f:
        f.write(data)
    plog(f"corpus: {len(data)/1e6:.1f} MB saved -> {out_path}")
    return data


# ── lightweight flesh sampler (trained notorch weights + ghost overlay) ───────
def flesh_sample(eng, karl, meta, prompt, n=48, temp=0.8, topk=20):
    ctx = eng.model.ctx
    out = list(karl.encode(prompt) or [0])
    V = eng.model.vocab_size
    for _ in range(n):
        logits = eng.forward_logits(out[-ctx:])              # trained flesh
        if meta is not None:                                  # ghost overlay (no destiny)
            bg = meta.query_bigram(out[-1], V)
            tg = (meta.query_trigram(out[-2], out[-1], V) if len(out) >= 2 else [0.0]*V)
            for i in range(V):
                logits[i] += 8.0 * bg[i] + 5.0 * tg[i]
        idx = sorted(range(V), key=lambda i: -logits[i])[:topk]
        mx = max(logits[i] for i in idx)
        ex = [(i, math.exp((logits[i]-mx)/temp)) for i in idx]
        s = sum(e for _, e in ex); r = random.random()*s; c = 0.0
        pick = idx[0]
        for i, e in ex:
            c += e
            if c >= r:
                pick = i; break
        out.append(pick)
    return karl.decode(out)


def flesh_magnitude(eng, ids, ctx, n=16):
    """Mean |flesh logit| — grows as Chuck trains; the ghost->flesh crossover signal."""
    random.seed(999)
    tot = 0.0
    for _ in range(n):
        i = random.randint(0, max(0, len(ids) - ctx - 2))
        row = eng.forward_logits(ids[i:i+ctx])
        tot += sum(abs(x) for x in row) / len(row)
    return tot / n


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    plog(f"\n{'='*70}\nnanoagi SUPERMILESTONE — notorch train  ({ts})\n{'='*70}")
    plog(f"backend={nn.NOTORCH_BACKEND} gpu={nn.NOTORCH_GPU} cfg={CFG}")

    data = build_corpus()

    # 2. KARL — learn merges on a bounded SAMPLE (BPE merges generalize), then FREEZE.
    # Learning on the full corpus is a CPU trap: 2048 merges × full-corpus pair rescans over
    # a Python token list. A few MB gives the same 2304-vocab; the merges tokenize the rest.
    karl = N.KARL(max_merges=CFG['merges'])
    km = CFG['karl_mb'] * 1_000_000
    sample = data[:km] if (km > 0 and km < len(data)) else data
    ids = karl.learn(sample, num_merges=CFG['merges'])
    V = karl.vocab_size
    plog(f"KARL: vocab={V} merges={len(karl.merges)} tokens={len(ids):,} "
         f"(learned on {len(sample)/1e6:.1f}MB sample, frozen)")

    # held-out val split
    split = int(len(ids) * 0.9)
    train_ids, val_ids = ids[:split], ids[split:]

    # metaweights (ghost) for the sampler overlay
    meta = N.MetaWeights(V, context_len=CFG['ctx'])
    meta.build(ids, window=4)

    # 3. model — 13.42M target at V=2304 / E384 H6 L5 ctx256 nc3 nr3
    nt_seed(42)
    m = NotorchNanoAGI(V, n_embd=CFG['embd'], n_head=CFG['head'], n_layer=CFG['layer'],
                       ctx=CFG['ctx'], n_content=CFG['content'], n_rrpram=CFG['rrpram'])
    eng = NotorchEngine(m, lr=CFG['lr'])
    dedup = sum(p.numel for p in eng.params)
    plog(f"model: {dedup:,} params (dedup)  ~{dedup*4/1e6:.1f} MB fp32")

    # 4. train loop + paper-log
    ctx, steps = CFG['ctx'], CFG['steps']
    plog(f"\nstep | train | val | |flesh| | t  (val every {CFG['val_every']})")
    random.seed(7)
    t0 = time.time()
    run = []
    pre_flesh = flesh_magnitude(eng, train_ids, ctx)
    plog(f"  pre-train |flesh|={pre_flesh:.4f}")
    for s in range(1, steps + 1):
        i = random.randint(0, max(0, len(train_ids) - ctx - 2))
        x = train_ids[i:i+ctx]; y = train_ids[i+1:i+ctx+1]
        if len(x) < ctx or len(y) < ctx:
            continue
        run.append(eng.step(x, y))
        if s % CFG['val_every'] == 0 or s == steps:
            tr = sum(run[-CFG['val_every']:]) / len(run[-CFG['val_every']:])
            vl = 0.0
            nv = min(20, max(1, len(val_ids)//ctx))
            for _ in range(nv):
                j = random.randint(0, max(0, len(val_ids)-ctx-2))
                if j+ctx+1 <= len(val_ids):
                    vl += eng.step(val_ids[j:j+ctx], val_ids[j+1:j+ctx+1], update=False)
            vl /= nv
            fm = flesh_magnitude(eng, train_ids, ctx)
            dt = time.time() - t0
            plog(f"  {s:6d} | {tr:.4f} | {vl:.4f} | {fm:.4f} | {dt:.0f}s "
                 f"({s/dt:.2f} st/s, gpu_dispatch={nn.gpu_dispatch_count()})")
            if s % (CFG['val_every']*4) == 0 or s == steps:
                samp = flesh_sample(eng, karl, meta, "the model", n=40)
                plog(f"     flesh> {samp[:200]!r}")

    # 5. save weights + manifest (<80MB target)
    eng.save(CFG['ckpt'])
    N._write_manifest(CFG['ckpt'], m, karl, steps)
    sz = os.path.getsize(CFG['ckpt']) / 1e6
    elapsed = time.time() - t0
    plog(f"\nSAVED {CFG['ckpt']} ({sz:.1f} MB) + manifest  [{elapsed:.0f}s total]")
    plog(f"crossover: |flesh| {pre_flesh:.4f} -> {flesh_magnitude(eng, train_ids, ctx):.4f}")
    plog(f"ghost>  {N.continue_phrase('the model', karl, meta, N.NanoAGI(vocab_size=V, context_len=CFG['ctx'], n_embd=CFG['embd'], n_head=CFG['head'], n_layer=CFG['layer'], n_content=CFG['content'], n_rrpram=CFG['rrpram']), max_tokens=40)[:200]!r}" if dedup < 2_000_000 else "ghost> (skipped: Python model too large to instantiate at scale)")
    plog(f"peak RSS: {rss_mb():.0f} MB | backend={nn.NOTORCH_BACKEND} gpu={nn.NOTORCH_GPU}")
    plog(f"{'='*70}\nDONE\n{'='*70}")


if __name__ == '__main__':
    main()

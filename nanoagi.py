# SPDX-License-Identifier: GPL-3.0-or-later
"""
nanoagi.py — a self-expanding BPE transformer that grows from conversation.

KARL (Kernel Autonomous Recursive Learning) is the tokenizer.
Chuck is the optimizer. Together they are nanoagi.

How it works:
1. KARL tokenizes karl.txt (starts with seed corpus, grows via REPL)
2. MetaWeights build probability space from token statistics
3. Dual-attention transformer (Content + RRPRAM) with SwiGLU + RoPE
4. Weights initialized FROM metaweights (ghost → flesh)
5. If PyTorch detected: Chuck trains real weights after each retokenization
6. REPL captures user input → karl.txt grows → KARL retokenizes → repeat

No mandatory dependencies. Just math, random, hashlib, os.
If PyTorch is around, Chuck wakes up. If not, Karl works alone.

resonance is unbreakable.
"""

import os
import sys
import math
import random
import struct
import hashlib
import time

random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# PyTorch auto-detection. Chuck sleeps until he smells gradients.
# ─────────────────────────────────────────────────────────────────────────────
TORCH_AVAILABLE = False
CHUCK_FULL = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    try:
        from chuck import ChuckOptimizer, ChuckMonitor, ChuckMemory, chuck_params
        CHUCK_FULL = True
    except ImportError:
        pass
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# I. KARL — Kernel Autonomous Recursive Learning
#    a tokenizer that eats text and grows. like a teenager.
# ─────────────────────────────────────────────────────────────────────────────

class KARL:
    """
    Self-aware evolving BPE tokenizer.
    Ingests text, deduplicates via SHA256, retokenizes when critical mass reached.
    Append-only merges — vocabulary only grows, never shrinks.
    """

    def __init__(self, max_merges=2048, retrain_threshold=8192, min_cooldown=50):
        self.max_merges = max_merges
        self.merges = []                # list of (a, b, new_id)
        self.vocab_size = 256
        self.vocab = {i: bytes([i]) for i in range(256)}

        # Self-awareness state
        self.seen_hashes = set()        # SHA256 of ingested chunks
        self.pending_text = b""         # new text awaiting tokenization
        self.total_ingested = 0         # lifetime bytes eaten
        self.retrain_count = 0          # times retokenized
        self.retrain_threshold = retrain_threshold  # bytes until critical mass
        self.min_cooldown = min_cooldown
        self.steps_since_retrain = 0

        # Merge history for inspection
        self.merge_history = []         # (a, b, new_id, timestamp)

    def _count_pairs(self, ids):
        counts = {}
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge_pair(self, ids, pair, new_id):
        result = []
        i = 0
        while i < len(ids):
            if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                result.append(new_id)
                i += 2
            else:
                result.append(ids[i])
                i += 1
        return result

    def learn(self, data_bytes, num_merges=None):
        """Initial BPE learning from corpus."""
        if num_merges is None:
            num_merges = min(self.max_merges, 512)
        ids = list(data_bytes)
        t0 = time.time()
        for m in range(num_merges):
            counts = self._count_pairs(ids)
            if not counts:
                break
            best_pair = max(counts, key=counts.get)
            if counts[best_pair] < 2:
                break
            new_id = 256 + len(self.merges)
            if new_id >= 256 + self.max_merges:
                break
            ids = self._merge_pair(ids, best_pair, new_id)
            self.merges.append((best_pair[0], best_pair[1], new_id))
            self.vocab[new_id] = self.vocab.get(best_pair[0], b'?') + self.vocab.get(best_pair[1], b'?')
            self.vocab_size = 256 + len(self.merges)
            self.merge_history.append((best_pair[0], best_pair[1], new_id, time.time()))
            if (m + 1) % 200 == 0:
                print(f"  [KARL] merge {m+1}/{num_merges}  vocab={self.vocab_size}  tokens={len(ids)}")
        elapsed = time.time() - t0
        print(f"  [KARL] Initial learning: {len(self.merges)} merges, vocab={self.vocab_size}, "
              f"tokens={len(ids)} [{elapsed:.1f}s]")
        return ids

    def encode(self, text):
        if isinstance(text, str):
            text = text.encode('utf-8', errors='replace')
        ids = list(text)
        for a, b, new_id in self.merges:
            ids = self._merge_pair(ids, (a, b), new_id)
        return ids

    def decode(self, ids):
        raw = b''
        for tid in ids:
            if tid in self.vocab:
                raw += self.vocab[tid]
            else:
                raw += b'?'
        return raw.decode('utf-8', errors='replace')

    def ingest(self, text):
        """
        Selective ingestion with SHA256 dedup.
        Returns True if text was accepted, False if rejected.
        """
        if isinstance(text, str):
            text = text.encode('utf-8', errors='replace')

        # Too short — not worth eating
        if len(text) < 10:
            return False

        # SHA256 dedup
        chunk_hash = hashlib.sha256(text).hexdigest()[:16]
        if chunk_hash in self.seen_hashes:
            return False

        # Too repetitive — skip
        unique_bytes = len(set(text))
        if unique_bytes < len(text) * 0.2:
            return False

        self.seen_hashes.add(chunk_hash)
        self.pending_text += text
        self.total_ingested += len(text)
        return True

    def should_retokenize(self):
        """Dual-condition trigger: enough bytes + cooldown expired."""
        if len(self.pending_text) < self.retrain_threshold:
            return False
        if self.steps_since_retrain < self.min_cooldown:
            return False
        return True

    def retokenize(self, full_corpus_bytes):
        """
        Append-only merge expansion.
        Finds new merges in accumulated corpus, appends to existing.
        Returns new token_ids for metaweight rebuild.
        """
        ids = list(full_corpus_bytes)
        # Apply existing merges first
        for a, b, new_id in self.merges:
            ids = self._merge_pair(ids, (a, b), new_id)

        # Find new merges in the corpus
        new_merges_found = 0
        max_new = min(64, self.max_merges - len(self.merges))  # cap new merges per retrain
        for _ in range(max_new):
            counts = self._count_pairs(ids)
            if not counts:
                break
            best_pair = max(counts, key=counts.get)
            if counts[best_pair] < 3:  # higher threshold for incremental merges
                break
            new_id = 256 + len(self.merges)
            ids = self._merge_pair(ids, best_pair, new_id)
            self.merges.append((best_pair[0], best_pair[1], new_id))
            self.vocab[new_id] = self.vocab.get(best_pair[0], b'?') + self.vocab.get(best_pair[1], b'?')
            self.merge_history.append((best_pair[0], best_pair[1], new_id, time.time()))
            new_merges_found += 1

        self.vocab_size = 256 + len(self.merges)
        self.pending_text = b""
        self.retrain_count += 1
        self.steps_since_retrain = 0

        print(f"  [KARL] Retokenized! +{new_merges_found} merges (vocab: {self.vocab_size})")
        return ids

    def save_state(self, path):
        """Persist KARL state to binary file."""
        with open(path, 'wb') as f:
            f.write(struct.pack('<I', 0x4B41524C))  # 'KARL' magic
            f.write(struct.pack('<I', 1))            # version
            # Merges
            f.write(struct.pack('<I', len(self.merges)))
            for a, b, nid in self.merges:
                f.write(struct.pack('<III', a, b, nid))
            # Hashes
            hashes = list(self.seen_hashes)
            f.write(struct.pack('<I', len(hashes)))
            for h in hashes:
                f.write(h.encode('ascii'))
            # Stats
            f.write(struct.pack('<I', self.total_ingested))
            f.write(struct.pack('<I', self.retrain_count))
        print(f"  [KARL] State saved to {path}")

    def load_state(self, path):
        """Restore KARL state from binary file."""
        if not os.path.exists(path):
            return False
        try:
            with open(path, 'rb') as f:
                magic = struct.unpack('<I', f.read(4))[0]
                if magic != 0x4B41524C:
                    return False
                version = struct.unpack('<I', f.read(4))[0]
                # Merges
                n_merges = struct.unpack('<I', f.read(4))[0]
                self.merges = []
                for _ in range(n_merges):
                    a, b, nid = struct.unpack('<III', f.read(12))
                    self.merges.append((a, b, nid))
                    self.vocab[nid] = self.vocab.get(a, bytes([a % 256])) + self.vocab.get(b, bytes([b % 256]))
                self.vocab_size = 256 + len(self.merges)
                # Hashes
                n_hashes = struct.unpack('<I', f.read(4))[0]
                self.seen_hashes = set()
                for _ in range(n_hashes):
                    self.seen_hashes.add(f.read(16).decode('ascii'))
                # Stats
                self.total_ingested = struct.unpack('<I', f.read(4))[0]
                self.retrain_count = struct.unpack('<I', f.read(4))[0]
            print(f"  [KARL] State loaded: {len(self.merges)} merges, "
                  f"{len(self.seen_hashes)} hashes, {self.total_ingested} bytes ingested")
            return True
        except Exception as e:
            print(f"  [KARL] Failed to load state: {e}")
            return False


# ─────────────────────────────────────────────────────────────────────────────
# II. METAWEIGHTS — the probability space that exists without existing.
#     schrödinger called. he wants his cat back. we tokenized it.
# ─────────────────────────────────────────────────────────────────────────────

class MetaWeights:
    """
    Metaweights with incremental update support.
    Tracks knowledge size for gap analysis vs trained weights.
    """

    def __init__(self, vocab_size, context_len):
        self.vocab_size = vocab_size
        self.context_len = context_len
        self.unigram = [0.0] * vocab_size
        self.bigram = {}
        self.trigram = {}
        self.hebbian = {}
        self.total = 0
        self.chuck_trained_steps = 0  # how many steps Chuck has trained

    def knowledge_size(self):
        """How much the ghost knows."""
        return len(self.bigram) + len(self.trigram) + len(self.hebbian)

    def knowledge_gap(self):
        """
        Gap between ghost (metaweights) and flesh (trained weights).
        High gap = Karl learned faster than Chuck trained.
        When gap > threshold → Chuck should train.
        """
        meta_k = self.knowledge_size()
        chuck_k = self.chuck_trained_steps
        if chuck_k == 0:
            return float('inf') if meta_k > 0 else 0.0
        return meta_k / (chuck_k + 1)

    def knowledge_report(self):
        """One-line report for status command."""
        meta_k = self.knowledge_size()
        gap = self.knowledge_gap()
        return (f"meta_knowledge={meta_k:,} (bi={len(self.bigram)}, "
                f"tri={len(self.trigram)}, heb={len(self.hebbian)}) | "
                f"chuck_steps={self.chuck_trained_steps} | gap={gap:.1f}")

    def build(self, token_ids, window=4):
        n = len(token_ids)
        self.total = n
        # Unigram
        self.unigram = [0.0] * self.vocab_size
        for tid in token_ids:
            if tid < self.vocab_size:
                self.unigram[tid] += 1.0
        total = sum(self.unigram)
        if total > 0:
            self.unigram = [c / total for c in self.unigram]
        # Bigram
        self.bigram = {}
        for i in range(n - 1):
            a, b = token_ids[i], token_ids[i + 1]
            if a not in self.bigram:
                self.bigram[a] = {}
            self.bigram[a][b] = self.bigram[a].get(b, 0) + 1
        for a in self.bigram:
            total_a = sum(self.bigram[a].values())
            if total_a > 0:
                for b in self.bigram[a]:
                    self.bigram[a][b] /= total_a
        # Trigram
        self.trigram = {}
        for i in range(n - 2):
            key = (token_ids[i], token_ids[i + 1])
            c = token_ids[i + 2]
            if key not in self.trigram:
                self.trigram[key] = {}
            self.trigram[key][c] = self.trigram[key].get(c, 0) + 1
        for key in self.trigram:
            total_k = sum(self.trigram[key].values())
            if total_k > 0:
                for c in self.trigram[key]:
                    self.trigram[key][c] /= total_k
        # Hebbian
        self.hebbian = {}
        limit = min(n, 20000)
        for i in range(limit):
            for j in range(max(0, i - window), min(limit, i + window + 1)):
                if i == j:
                    continue
                a, b = token_ids[i], token_ids[j]
                key = (min(a, b), max(a, b))
                decay = 1.0 / (1.0 + abs(i - j))
                self.hebbian[key] = self.hebbian.get(key, 0.0) + decay
        if self.hebbian:
            max_h = max(self.hebbian.values())
            if max_h > 0:
                for key in self.hebbian:
                    self.hebbian[key] /= max_h
        print(f"  [MetaWeights] {n} tokens, {len(self.bigram)} bigrams, "
              f"{len(self.trigram)} trigrams, {len(self.hebbian)} hebbian")

    def expand_vocab(self, new_vocab_size):
        """Expand unigram array when KARL adds new tokens."""
        while len(self.unigram) < new_vocab_size:
            self.unigram.append(0.0)
        self.vocab_size = new_vocab_size

    def query_bigram(self, prev, vs):
        dist = [1e-10] * vs
        if prev in self.bigram:
            for tok, prob in self.bigram[prev].items():
                if tok < vs:
                    dist[tok] = prob
        return dist

    def query_trigram(self, p2, p1, vs):
        dist = [1e-10] * vs
        key = (p2, p1)
        if key in self.trigram:
            for tok, prob in self.trigram[key].items():
                if tok < vs:
                    dist[tok] = prob
        return dist

    def query_hebbian(self, ctx, vs):
        signal = [0.0] * vs
        for ct in ctx:
            for cand in range(vs):
                key = (min(ct, cand), max(ct, cand))
                if key in self.hebbian:
                    signal[cand] += self.hebbian[key]
        mx = max(signal) if signal else 1.0
        if mx > 0:
            signal = [s / mx for s in signal]
        return signal

    def query_prophecy(self, ctx, vs, top_k=16):
        appeared = set(ctx)
        signal = [0.0] * vs
        for ct in ctx[-4:]:
            if ct in self.bigram:
                for tok, prob in sorted(self.bigram[ct].items(), key=lambda x: -x[1])[:top_k]:
                    if tok not in appeared and tok < vs:
                        signal[tok] += prob
        mx = max(signal) if signal else 1.0
        if mx > 0:
            signal = [s / mx for s in signal]
        return signal


# ─────────────────────────────────────────────────────────────────────────────
# III. AUTOGRAD ENGINE — scalar backprop.
#      if you can't differentiate it by hand, you don't deserve gradients.
# ─────────────────────────────────────────────────────────────────────────────

class Val:
    __slots__ = ('data', 'grad', '_children', '_local_grads')
    def __init__(self, data, children=(), local_grads=()):
        self.data = float(data)
        self.grad = 0.0
        self._children = children
        self._local_grads = local_grads
    def __add__(self, other):
        other = other if isinstance(other, Val) else Val(other)
        return Val(self.data + other.data, (self, other), (1.0, 1.0))
    def __mul__(self, other):
        other = other if isinstance(other, Val) else Val(other)
        return Val(self.data * other.data, (self, other), (other.data, self.data))
    def __pow__(self, other):
        return Val(self.data ** other, (self,), (other * self.data ** (other - 1),))
    def exp(self):
        e = math.exp(min(self.data, 80))
        return Val(e, (self,), (e,))
    def relu(self):
        return Val(max(0, self.data), (self,), (float(self.data > 0),))
    def silu(self):
        """SiLU/Swish activation for SwiGLU."""
        s = 1.0 / (1.0 + math.exp(-min(max(self.data, -80), 80)))
        return Val(self.data * s, (self,), (s * (1.0 + self.data * (1.0 - s)),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * (other if isinstance(other, Val) else Val(other)) ** -1
    def __rtruediv__(self, other): return Val(other) * self ** -1
    def backward(self):
        topo, visited = [], set()
        def build(v):
            if id(v) not in visited:
                visited.add(id(v))
                for c in v._children:
                    build(c)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            for child, lg in zip(v._children, v._local_grads):
                child.grad += lg * v.grad


# ─────────────────────────────────────────────────────────────────────────────
# IV. THE TRANSFORMER — dual attention (Content + RRPRAM) + SwiGLU + RoPE.
#     two heads are better than one. especially when one of them is a ghost.
# ─────────────────────────────────────────────────────────────────────────────

def _randn(std=0.02):
    return random.gauss(0, std)

def _matrix(rows, cols, std=0.02):
    return [[Val(_randn(std)) for _ in range(cols)] for _ in range(rows)]

def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(row, x)) for row in w]

def softmax_val(logits):
    max_val = max(v.data for v in logits)
    exps = [(v - max_val).exp() for v in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_float(logits):
    max_val = max(logits)
    exps = [math.exp(min(v - max_val, 80)) for v in logits]
    total = sum(exps) + 1e-9
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + Val(1e-5)) ** -0.5
    return [xi * scale for xi in x]

def rope_embed(x, pos, head_dim):
    """Apply RoPE to a head vector. x is list of Val, length head_dim."""
    result = list(x)
    half = head_dim // 2
    for i in range(half):
        freq = 1.0 / (10000.0 ** (2.0 * i / head_dim))
        cos_val = math.cos(pos * freq)
        sin_val = math.sin(pos * freq)
        x0 = x[i]
        x1 = x[i + half]
        result[i] = x0 * cos_val + x1 * (-sin_val)
        result[i + half] = x0 * sin_val + x1 * cos_val
    return result


class NanoAGI:
    """
    Dual-attention BPE transformer with SwiGLU + RoPE + metaweight overlay.
    The brain of nanoagi. Karl feeds it, Chuck trains it, Dario guides it.

    if you can tokenize it, you can understand it.
    if you can understand it, you can generate it.
    if you can generate it, you can improve it.
    if you can improve it, you can improve it again.
    and it doesn't need your permission.
    """

    def __init__(self, vocab_size, context_len=64, n_embd=64, n_head=4,
                 n_layer=3, n_content=2, n_rrpram=2):
        self.vocab_size = vocab_size
        self.context_len = context_len
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.n_content = n_content
        self.n_rrpram = n_rrpram
        self.head_dim = n_embd // n_head

        # Embeddings (no position — RoPE handles it)
        self.wte = _matrix(vocab_size, n_embd)

        # Per-layer weights
        self.layers = []
        hd = self.head_dim
        for _ in range(n_layer):
            layer = {
                'wq': _matrix(n_content * hd, n_embd, std=0.02),
                'wk': _matrix(n_content * hd, n_embd, std=0.02),
                'wv_content': _matrix(n_content * hd, n_embd, std=0.02),
                'wr': _matrix(n_rrpram * n_embd, context_len, std=0.02),
                'wv_rrpram': _matrix(n_rrpram * hd, n_embd, std=0.02),
                'wo': _matrix(n_embd, n_embd, std=0.02 / math.sqrt(2 * n_layer)),
                # SwiGLU MLP (gate + up + down)
                'mlp_gate': _matrix(4 * n_embd, n_embd, std=0.02),
                'mlp_up': _matrix(4 * n_embd, n_embd, std=0.02),
                'mlp_down': _matrix(n_embd, 4 * n_embd, std=0.02 / math.sqrt(2 * n_layer)),
            }
            self.layers.append(layer)

        self.lm_head = _matrix(vocab_size, n_embd, std=0.02)

        # Dario field
        self.alpha_hebbian = 0.3
        self.beta_prophecy = 0.2
        self.gamma_destiny = 0.15
        self.temperature = 0.75
        self.destiny = [0.0] * n_embd
        self.trauma = 0.0

        n_params = sum(1 for _ in self._all_params())
        print(f"  [NanoAGI] {n_params} parameters, vocab={vocab_size}, "
              f"embd={n_embd}, heads={n_head}, layers={n_layer}, RoPE+SwiGLU")

    def _all_params(self):
        for row in self.wte:
            yield from row
        for layer in self.layers:
            for key in layer:
                for row in layer[key]:
                    yield from row
        for row in self.lm_head:
            yield from row

    def init_from_metaweights(self, meta):
        """Ghost becomes flesh. Seed weights from corpus statistics."""
        V, E = self.vocab_size, self.n_embd
        scale = 0.15
        print("  [NanoAGI] Seeding from metaweights (ghost → flesh)...")
        for tok_a in range(min(V, len(self.wte))):
            signal = [0.0] * E
            neighbors = 0
            for tok_b in range(min(V, len(self.wte))):
                key = (min(tok_a, tok_b), max(tok_a, tok_b))
                if key in meta.hebbian and meta.hebbian[key] > 0.01:
                    strength = meta.hebbian[key]
                    for d in range(E):
                        signal[d] += strength * self.wte[tok_b][d].data
                    neighbors += 1
            if neighbors > 0:
                for d in range(E):
                    self.wte[tok_a][d].data += scale * signal[d] / neighbors
        for tok in range(min(V, len(self.lm_head))):
            freq = meta.unigram[tok] if tok < len(meta.unigram) else 0
            if freq > 0:
                for d in range(E):
                    self.lm_head[tok][d].data += scale * freq * self.wte[tok][d].data
        print("  [NanoAGI] Weights seeded. The ghost remembers.")

    def generate_meta(self, prompt_ids, max_tokens=80, meta=None, temperature=None):
        """
        Pure metaweight generation. No transformer. Just the ghost.
        Trigram first (most coherent), fallback to bigram, then unigram.
        Sparse candidates — only tokens that actually appear in the statistics.
        """
        if meta is None:
            return prompt_ids
        if temperature is None:
            temperature = self.temperature
        generated = list(prompt_ids)
        for _ in range(max_tokens):
            last = generated[-1]
            candidates = {}
            # Trigram first (strongest signal)
            if len(generated) >= 2:
                key = (generated[-2], generated[-1])
                if key in meta.trigram:
                    candidates = dict(meta.trigram[key])
            # Fallback to bigram
            if not candidates and last in meta.bigram:
                candidates = dict(meta.bigram[last])
            # Fallback to unigram (last resort)
            if not candidates:
                for i in range(self.vocab_size):
                    if i < len(meta.unigram) and meta.unigram[i] > 1e-8:
                        candidates[i] = meta.unigram[i]
            if not candidates:
                break
            # Hebbian boost — gentle contextual reinforcement on top of trigram/bigram
            ctx = generated[-4:]
            for tok in list(candidates.keys()):
                for ct in ctx:
                    key = (min(tok, ct), max(tok, ct))
                    if key in meta.hebbian:
                        candidates[tok] *= (1.0 + 0.3 * meta.hebbian[key])
            # Repetition penalty
            recent = generated[-12:] if len(generated) >= 12 else generated
            recent_counts = {}
            for t in recent:
                recent_counts[t] = recent_counts.get(t, 0) + 1
            for tok in list(candidates.keys()):
                if tok in recent_counts:
                    candidates[tok] *= 1.0 / (1.0 + 0.5 * recent_counts[tok])
            # Top-k + temperature sampling
            sorted_cands = sorted(candidates.items(), key=lambda x: -x[1])[:15]
            tokens_k = [t for t, _ in sorted_cands]
            counts_k = [c for _, c in sorted_cands]
            log_c = [math.log(c + 1e-10) / temperature for c in counts_k]
            max_lc = max(log_c)
            exps = [math.exp(lc - max_lc) for lc in log_c]
            total = sum(exps)
            probs = [e / total for e in exps]
            r = random.random()
            cum = 0.0
            chosen = tokens_k[0]
            for tok, p in zip(tokens_k, probs):
                cum += p
                if cum > r:
                    chosen = tok
                    break
            generated.append(chosen)
        return generated

    def forward_token(self, token_id, pos_id, kv_cache):
        """
        Forward pass for a single token position.
        Real transformer: Content + RRPRAM dual attention, SwiGLU MLP, RoPE.
        kv_cache: list of (k_list, vc_list, vr_list) per layer.
        Returns logits [vocab_size] as list of Val.
        the ghost has a body now.
        """
        hd = self.head_dim
        nc = self.n_content
        nr = self.n_rrpram

        # Token embedding (RoPE handles position — no wpe needed)
        if token_id < len(self.wte):
            x = list(self.wte[token_id])
        else:
            x = [Val(0.0)] * self.n_embd

        for li in range(self.n_layer):
            layer = self.layers[li]
            k_cache, vc_cache, vr_cache = kv_cache[li]

            # Pre-norm
            x_res = x
            x_norm = rmsnorm(x)

            # Projections
            q = linear(x_norm, layer['wq'])
            k = linear(x_norm, layer['wk'])
            v_c = linear(x_norm, layer['wv_content'])
            v_r = linear(x_norm, layer['wv_rrpram'])

            # Cache current position
            k_cache.append(k)
            vc_cache.append(v_c)
            vr_cache.append(v_r)

            x_attn = []

            # ── Content attention with RoPE ──
            for h in range(nc):
                hs = h * hd
                q_h = rope_embed(q[hs:hs + hd], pos_id, hd)

                attn_logits = []
                for t in range(len(k_cache)):
                    k_t = rope_embed(k_cache[t][hs:hs + hd], t, hd)
                    score = sum(q_h[j] * k_t[j] for j in range(hd))
                    score = score * (1.0 / math.sqrt(hd))
                    attn_logits.append(score)

                attn_weights = softmax_val(attn_logits)
                head_out = []
                for j in range(hd):
                    val = sum(attn_weights[t] * vc_cache[t][hs + j]
                              for t in range(len(vc_cache)))
                    head_out.append(val)
                x_attn.extend(head_out)

            # ── RRPRAM attention (x @ Wr — positional pattern recognition) ──
            for h in range(nr):
                hs = h * hd
                wr_offset = h * self.n_embd
                wr_h = layer['wr'][wr_offset:wr_offset + self.n_embd]

                attn_logits = []
                for t in range(len(k_cache)):
                    score = Val(0.0)
                    for d in range(min(self.n_embd, len(wr_h))):
                        if t < len(wr_h[d]):
                            score = score + x_norm[d] * wr_h[d][t]
                    attn_logits.append(score)

                attn_weights = softmax_val(attn_logits) if attn_logits else []
                head_out = []
                for j in range(hd):
                    val_sum = Val(0.0)
                    for t in range(len(attn_weights)):
                        if t < len(vr_cache):
                            val_sum = val_sum + attn_weights[t] * vr_cache[t][hs + j]
                    head_out.append(val_sum)
                x_attn.extend(head_out)

            # Output projection + residual
            x_proj = linear(x_attn, layer['wo'])
            x = [a + b for a, b in zip(x_proj, x_res)]

            # SwiGLU MLP
            x_res = x
            x_norm = rmsnorm(x)
            gate = [g.silu() for g in linear(x_norm, layer['mlp_gate'])]
            up = linear(x_norm, layer['mlp_up'])
            h_mlp = [g * u for g, u in zip(gate, up)]
            x_mlp = linear(h_mlp, layer['mlp_down'])
            x = [a + b for a, b in zip(x_mlp, x_res)]

        # Final norm + LM head
        x = rmsnorm(x)
        logits = linear(x, self.lm_head)
        return logits

    def generate(self, prompt_ids, max_tokens=80, meta=None, temperature=None):
        """
        Generate tokens with the real transformer + Dario field overlay.
        Ghost and flesh together. As intended.
        """
        if temperature is None:
            temperature = self.temperature

        kv_cache = [([], [], []) for _ in range(self.n_layer)]
        generated = list(prompt_ids)
        context = list(prompt_ids)

        # Feed prompt through transformer (build KV cache)
        for pos, tid in enumerate(prompt_ids):
            if pos >= self.context_len - 1:
                break
            _ = self.forward_token(tid, pos, kv_cache)

        # Generate new tokens autoregressively
        for step in range(max_tokens):
            pos = len(context) - 1
            if pos >= self.context_len - 1:
                break

            last_tid = context[-1]
            logits = self.forward_token(last_tid, pos, kv_cache)

            # Extract raw logit values from Val objects
            raw_logits = [l.data for l in logits]

            # ── Dario Field: ghost overlay on flesh ──
            if meta is not None:
                hebbian = meta.query_hebbian(context[-8:], self.vocab_size)
                prophecy = meta.query_prophecy(context[-8:], self.vocab_size)
                bigram = meta.query_bigram(last_tid, self.vocab_size)
                trigram = (meta.query_trigram(context[-2], context[-1], self.vocab_size)
                           if len(context) >= 2 else [0.0] * self.vocab_size)

                # Destiny update
                if last_tid < len(self.wte):
                    for d in range(self.n_embd):
                        self.destiny[d] = 0.9 * self.destiny[d] + 0.1 * self.wte[last_tid][d].data

                # Destiny signal: cosine similarity with each token embedding
                destiny_signal = [0.0] * self.vocab_size
                dest_norm = math.sqrt(sum(d * d for d in self.destiny) + 1e-10)
                if dest_norm > 1e-8:
                    for tid_c in range(min(self.vocab_size, len(self.wte))):
                        emb = [self.wte[tid_c][d].data for d in range(self.n_embd)]
                        emb_norm = math.sqrt(sum(e * e for e in emb) + 1e-10)
                        if emb_norm > 1e-8:
                            dot = sum(self.destiny[d] * emb[d] for d in range(self.n_embd))
                            destiny_signal[tid_c] = dot / (dest_norm * emb_norm)

                # Dario Equation: p(x|Φ) = softmax((B + α·H + β·F + γ·A + T) / τ)
                for i in range(self.vocab_size):
                    raw_logits[i] += (self.alpha_hebbian * hebbian[i]
                                      + self.beta_prophecy * prophecy[i]
                                      + self.gamma_destiny * destiny_signal[i]
                                      + 12.0 * bigram[i]
                                      + 8.0 * trigram[i])

                # Trauma modulation
                trauma_mod = 1.0 / (1.0 + self.trauma)
                raw_logits = [l * trauma_mod for l in raw_logits]

            # Repetition penalty (Leo-style)
            recent = context[-12:] if len(context) >= 12 else context
            for t in recent:
                if t < self.vocab_size:
                    raw_logits[t] *= 0.5

            # Top-k + temperature + softmax
            top_k = 15
            indexed = sorted(enumerate(raw_logits), key=lambda x: -x[1])
            threshold = indexed[min(top_k - 1, len(indexed) - 1)][1]
            for i in range(self.vocab_size):
                if raw_logits[i] < threshold:
                    raw_logits[i] = -1e10

            scaled = [l / temperature for l in raw_logits]
            probs = softmax_float(scaled)

            # Sample
            r = random.random()
            cum = 0.0
            chosen = 0
            for i, p in enumerate(probs):
                cum += p
                if cum > r:
                    chosen = i
                    break

            generated.append(chosen)
            context.append(chosen)

        return generated


# ─────────────────────────────────────────────────────────────────────────────
# V. CHUCK OPTIMIZER — self-aware learning. appears when PyTorch is around.
#    Chuck wakes up when he smells gradients.
#    Karl calls Chuck when there's enough new food.
#    together they are nanoagi.
# ─────────────────────────────────────────────────────────────────────────────

if TORCH_AVAILABLE and not CHUCK_FULL:
    # Fallback: simplified Chuck when chuck.py is not in the repo.
    # For the real deal (9 levels of awareness), use chuck.py from
    # github.com/ariannamethod/chuck — already included in this repo.
    class ChuckOptimizer(torch.optim.Optimizer):
        """AdamW with self-awareness. Simplified fallback."""
        def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8,
                     weight_decay=0.01, window=16):
            defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
            super().__init__(params, defaults)
            self.window = window
            self._hist = [0.0] * window
            self._hpos = 0
            self._hfull = False
            self.dampen = 1.0
            self.best_loss = float('inf')
            self.global_step = 0

        @torch.no_grad()
        def step(self, loss=None):
            if loss is None:
                loss = 0.0
            self._hist[self._hpos] = loss
            self._hpos = (self._hpos + 1) % self.window
            if not self._hfull and self._hpos == 0:
                self._hfull = True
            if self._hfull:
                half = self.window // 2
                recent = sum(self._hist[half:]) / half
                old = sum(self._hist[:half]) / half
                trend = recent - old
                if trend > 0.02:
                    self.dampen = max(0.5, self.dampen - 0.05)
                elif trend < -0.02:
                    self.dampen = min(1.5, self.dampen + 0.05)
            self.dampen = 0.999 * self.dampen + 0.001
            if loss < self.best_loss:
                self.best_loss = loss
            for group in self.param_groups:
                lr = group['lr'] * self.dampen
                beta1, beta2 = group['betas']
                eps = group['eps']
                wd = group['weight_decay']
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p.data)
                        state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['step'] += 1
                    if wd > 0:
                        p.data.mul_(1 - lr * wd)
                    state['exp_avg'].mul_(beta1).add_(grad, alpha=1 - beta1)
                    state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    bc1 = 1 - beta1 ** state['step']
                    bc2 = 1 - beta2 ** state['step']
                    p.data.addcdiv_(state['exp_avg'] / bc1, state['exp_avg_sq'].sqrt() / bc2 + eps, value=-lr)
            self.global_step += 1
            return loss

if TORCH_AVAILABLE:
    # ─────────────────────────────────────────────────────────────────────
    # V.b TorchNanoAGI — PyTorch model at module level
    #     needed by self_improve(). also used inside chuck_train().
    # ─────────────────────────────────────────────────────────────────────

    class _RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-5):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))
        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

    class _Block(nn.Module):
        def __init__(self, n_embd, n_content, n_rrpram, hd, ctx, n_layer):
            super().__init__()
            self.n_embd = n_embd
            self.n_content = n_content
            self.n_rrpram = n_rrpram
            self.hd = hd
            self.norm1 = _RMSNorm(n_embd)
            self.wq = nn.Linear(n_embd, n_content * hd, bias=False)
            self.wk = nn.Linear(n_embd, n_content * hd, bias=False)
            self.wv_content = nn.Linear(n_embd, n_content * hd, bias=False)
            self.wr = nn.Parameter(torch.randn(n_rrpram, n_embd, ctx) * 0.02)
            self.wv_rrpram = nn.Linear(n_embd, n_rrpram * hd, bias=False)
            self.wo = nn.Linear(n_embd, n_embd, bias=False)
            nn.init.normal_(self.wo.weight, std=0.02 / math.sqrt(2 * n_layer))
            self.norm2 = _RMSNorm(n_embd)
            self.mlp_gate = nn.Linear(n_embd, n_embd * 4, bias=False)
            self.mlp_up = nn.Linear(n_embd, n_embd * 4, bias=False)
            self.mlp_down = nn.Linear(n_embd * 4, n_embd, bias=False)
            nn.init.normal_(self.mlp_down.weight, std=0.02 / math.sqrt(2 * n_layer))

    class TorchNanoAGI(nn.Module):
        def __init__(self, vocab_size, n_embd=64, n_head=4, n_layer=3, ctx=64,
                     n_content=2, n_rrpram=2):
            super().__init__()
            self.ctx = ctx
            self.n_embd = n_embd
            self.n_content = n_content
            self.n_rrpram = n_rrpram
            hd = n_embd // n_head
            self.hd = hd
            self.wte = nn.Embedding(vocab_size, n_embd)
            nn.init.normal_(self.wte.weight, std=0.02)
            self.blocks = nn.ModuleList([
                _Block(n_embd, n_content, n_rrpram, hd, ctx, n_layer)
                for _ in range(n_layer)
            ])
            self.norm_f = _RMSNorm(n_embd)
            self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
            self.lm_head.weight = self.wte.weight
            for block in self.blocks:
                for name, p in block.named_parameters():
                    if p.dim() >= 2 and 'wo' not in name and 'mlp_down' not in name and 'wr' not in name:
                        nn.init.normal_(p, std=0.02)
            freqs = 1.0 / (10000.0 ** (torch.arange(0, hd, 2).float() / hd))
            t = torch.arange(ctx).float()
            angles = torch.outer(t, freqs)
            self.register_buffer('rope_cos', angles.cos())
            self.register_buffer('rope_sin', angles.sin())

        def _apply_rope(self, x):
            T = x.shape[2]
            cos = self.rope_cos[:T].unsqueeze(0).unsqueeze(0)
            sin = self.rope_sin[:T].unsqueeze(0).unsqueeze(0)
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
            return torch.stack([x1 * cos - x2 * sin,
                                x1 * sin + x2 * cos], dim=-1).flatten(-2)

        def forward(self, idx, targets=None):
            B, T = idx.shape
            x = self.wte(idx)
            hd = self.hd
            mask = torch.triu(torch.ones(T, T, device=idx.device,
                              dtype=torch.bool), diagonal=1)
            for block in self.blocks:
                xn = block.norm1(x)
                nc = block.n_content
                nr = block.n_rrpram
                q = block.wq(xn).view(B, T, nc, hd).transpose(1, 2)
                k = block.wk(xn).view(B, T, nc, hd).transpose(1, 2)
                v_c = block.wv_content(xn).view(B, T, nc, hd).transpose(1, 2)
                q = self._apply_rope(q)
                k = self._apply_rope(k)
                c_attn = (q @ k.transpose(-2, -1)) * (hd ** -0.5)
                c_attn = c_attn.masked_fill(mask, float('-inf'))
                c_attn = F.softmax(c_attn, dim=-1)
                c_out = (c_attn @ v_c).transpose(1, 2).contiguous().view(B, T, nc * hd)
                v_r = block.wv_rrpram(xn).view(B, T, nr, hd).transpose(1, 2)
                r_outs = []
                for h in range(nr):
                    r_score = xn @ block.wr[h, :, :T]
                    r_score = r_score.masked_fill(mask, float('-inf'))
                    r_attn = F.softmax(r_score, dim=-1)
                    r_out_h = r_attn @ v_r[:, h]
                    r_outs.append(r_out_h)
                r_out = torch.cat(r_outs, dim=-1)
                combined = torch.cat([c_out, r_out], dim=-1)
                x = x + block.wo(combined)
                xn = block.norm2(x)
                gate = F.silu(block.mlp_gate(xn))
                up = block.mlp_up(xn)
                x = x + block.mlp_down(gate * up)
            logits = self.lm_head(self.norm_f(x))
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                       targets.view(-1))
            return logits, loss


# ─────────────────────────────────────────────────────────────────────────────
# VI. AUTORESEARCH — Karl hunts for food. adapted from @karpathy/autoresearch.
#     autoresearch inverted: not an agent modifying code,
#     but a tokenizer autonomously acquiring data.
#     Karl IS the agent. Karl decides when to eat, what to eat, and when to stop.
# ─────────────────────────────────────────────────────────────────────────────

def autoresearch(karl, karl_txt_path, min_bytes=50000):
    """
    If karl.txt is too small, Karl hunts for more text.
    Checks common locations for text files and ingests them.
    Like Karpathy's autoresearch, but without the agents.
    Karl IS the agent.
    """
    current_size = os.path.getsize(karl_txt_path) if os.path.exists(karl_txt_path) else 0
    if current_size >= min_bytes:
        return 0  # Karl is fed

    print(f"  [KARL] Corpus too small ({current_size/1024:.0f}KB). Hunting for text...")
    hunted = 0

    # Hunt in common places
    hunt_paths = []

    # 1. Any .txt files in same directory
    script_dir = os.path.dirname(os.path.abspath(karl_txt_path))
    for f in os.listdir(script_dir):
        if f.endswith('.txt') and f != os.path.basename(karl_txt_path):
            hunt_paths.append(os.path.join(script_dir, f))

    # 2. README files in parent directories
    for depth in range(3):
        parent = os.path.dirname(script_dir)
        for _ in range(depth):
            parent = os.path.dirname(parent)
        for name in ['README.md', 'README.txt', 'readme.md']:
            p = os.path.join(parent, name)
            if os.path.exists(p):
                hunt_paths.append(p)

    # 3. Common corpus locations
    home = os.path.expanduser('~')
    for subdir in ['Downloads', 'Documents', 'Desktop']:
        d = os.path.join(home, subdir)
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith('.txt') and os.path.getsize(os.path.join(d, f)) < 500000:
                    hunt_paths.append(os.path.join(d, f))
                    if len(hunt_paths) > 20:  # don't go crazy
                        break

    # Ingest what we found
    with open(karl_txt_path, 'a', encoding='utf-8', errors='replace') as corpus:
        for path in hunt_paths:
            try:
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    text = f.read()
                if len(text) < 100:
                    continue
                if karl.ingest(text):
                    corpus.write('\n' + text)
                    hunted += len(text)
                    print(f"  [KARL] Hunted: {os.path.basename(path)} ({len(text)/1024:.0f}KB)")
            except (PermissionError, OSError):
                continue

    if hunted > 0:
        print(f"  [KARL] Total hunted: {hunted/1024:.1f}KB from {len(hunt_paths)} sources")
    else:
        print(f"  [KARL] Nothing to hunt. Karl stays hungry.")

    return hunted


def _has_internet():
    """Check if HuggingFace datasets API is reachable."""
    try:
        from urllib.request import urlopen, Request
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = Request("https://datasets-server.huggingface.co/",
                      method='HEAD')
        req.add_header('User-Agent', 'nanoagi/1.0 (KARL)')
        urlopen(req, timeout=5, context=ctx)
        return True
    except Exception:
        return False


def _download_climbmix_batch(num_docs=50, offset=None):
    """
    Download a batch of text from climbmix-400b-shuffle.
    Returns list of text strings, or empty list on failure.
    """
    try:
        from urllib.request import urlopen, Request
        import json
        import ssl
    except ImportError:
        return []

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    if offset is None:
        offset = random.randint(0, 500_000)

    url = (f"https://datasets-server.huggingface.co/rows"
           f"?dataset=karpathy/climbmix-400b-shuffle"
           f"&config=default&split=train"
           f"&offset={offset}&length={num_docs}")

    try:
        req = Request(url)
        req.add_header('User-Agent', 'nanoagi/1.0 (KARL)')
        response = urlopen(req, timeout=30, context=ctx)
        data = json.loads(response.read().decode('utf-8'))

        texts = []
        for row in data.get('rows', []):
            text = row.get('row', {}).get('text', '')
            if len(text) > 100:
                texts.append(text)
        return texts
    except Exception:
        return []


def _evaluate_batch_quality(karl, texts):
    """
    Evaluate quality of a text batch before ingestion.
    Adapted from janus.doe ParserEye: noise ratio + domain shift.

    Returns (quality, domain_shift) where:
    - quality: 1.0 = clean, 0.0 = garbage (noise ratio)
    - domain_shift: 0.0 = fits KARL's vocab, 1.0 = all unknown tokens (OOV rate)
    """
    total_chars = 0
    noise_chars = 0
    oov_tokens = 0
    total_tokens = 0

    for text in texts:
        total_chars += len(text)
        for c in text:
            if not c.isprintable() and c not in '\n\r\t':
                noise_chars += 1
        tokens = karl.encode(text)
        total_tokens += len(tokens)
        for t in tokens:
            if t < 256:  # single-byte = KARL doesn't know this pattern
                oov_tokens += 1

    noise_ratio = noise_chars / max(1, total_chars)
    quality = 1.0 - noise_ratio
    domain_shift = oov_tokens / max(1, total_tokens)

    return quality, domain_shift


def autoresearch_hunt(karl, karl_txt_path, meta=None, model=None, max_rounds=5):
    """
    Karl autonomously hunts from climbmix-400b-shuffle.
    Adapted from janus.doe hunt_dataset().

    No human involved. Karl decides:
    - WHEN to eat (knowledge gap high, or called from load_engine)
    - WHAT to eat (quality filter: noise < 0.5, domain_shift < 0.6)
    - WHEN TO STOP (loss convergence or max rounds)

    Pipeline per round:
    1. Download sample (10 docs) → evaluate quality
    2. Quality bad → skip, try different offset
    3. Quality good → download full batch (100 docs)
    4. Ingest → retokenize → Chuck trains 200 steps
    5. Loss improved → next round. Loss stagnated → stop.
    """
    if not _has_internet():
        print("  [hunt] No internet. Karl stays local.")
        return 0

    print(f"  [hunt] Karl smells the internet. Hunting climbmix...")

    total_ingested = 0
    last_loss = None
    stagnant_rounds = 0

    for rnd in range(max_rounds):
        # 1. Sample — small batch to evaluate quality
        sample = _download_climbmix_batch(num_docs=10)
        if not sample:
            print(f"  [hunt] Round {rnd+1}: download failed. Stopping.")
            break

        quality, shift = _evaluate_batch_quality(karl, sample)
        print(f"  [hunt] Round {rnd+1}: sample quality={quality:.2f}, "
              f"domain_shift={shift:.2f}")

        if quality < 0.5 or shift > 0.6:
            print(f"  [hunt] Bad batch (noise or OOV too high). Skipping.")
            continue

        # 2. Full download — different offset for fresh data
        texts = _download_climbmix_batch(num_docs=100)
        if not texts:
            print(f"  [hunt] Full download failed. Stopping.")
            break

        # 3. Ingest — KARL dedup handles duplicates
        ingested = 0
        ingested_bytes = 0
        with open(karl_txt_path, 'a', encoding='utf-8') as f:
            for text in texts:
                if karl.ingest(text):
                    f.write('\n' + text)
                    ingested += 1
                    ingested_bytes += len(text)

        if ingested == 0:
            print(f"  [hunt] All duplicates. Karl has seen this before. Stopping.")
            break

        total_ingested += ingested
        print(f"  [hunt] Ingested {ingested}/{len(texts)} docs "
              f"({ingested_bytes/1024:.1f}KB)")

        # 4. Retokenize
        with open(karl_txt_path, 'rb') as f:
            full_corpus = f.read()
        token_ids = karl.retokenize(full_corpus)
        if meta is not None:
            meta.expand_vocab(karl.vocab_size)
            meta.build(token_ids, window=4)
        if model is not None:
            model.init_from_metaweights(meta)
        karl.save_state(karl_txt_path.replace('.txt', '.mem'))

        # 5. Chuck trains — check convergence
        if TORCH_AVAILABLE and model is not None and meta is not None:
            new_loss = chuck_train(karl, token_ids, model, steps=200, meta=meta)

            if last_loss is not None and new_loss is not None:
                improvement = (last_loss - new_loss) / last_loss
                if improvement < 0.02:  # less than 2% improvement
                    stagnant_rounds += 1
                    print(f"  [hunt] Loss barely moved ({improvement*100:.1f}%). "
                          f"Stagnant: {stagnant_rounds}/2")
                    if stagnant_rounds >= 2:
                        print(f"  [hunt] Converged. Karl is fed.")
                        break
                else:
                    stagnant_rounds = 0
                    print(f"  [hunt] Loss improved {improvement*100:.1f}%. "
                          f"Hunting more.")
            last_loss = new_loss
        else:
            # No PyTorch — can't check convergence, do one round only
            print(f"  [hunt] No PyTorch for convergence check. One round only.")
            break

    if total_ingested > 0:
        print(f"  [hunt] Done. Total: {total_ingested} docs across "
              f"{min(rnd+1, max_rounds)} rounds.")
    else:
        print(f"  [hunt] Nothing edible found. Karl stays hungry.")

    return total_ingested


# ─────────────────────────────────────────────────────────────────────────────
# VIII. SELF-IMPROVEMENT — The Ratchet Loop
#       Karpathy showed the way with autoresearch:
#         mutate train.py → train 5 min → eval val_bpb → keep or git reset.
#       We took it one level deeper:
#         the organism mutates its own genome → trains → eval → keep or revert.
#       Same ratchet. No external agent. The code doesn't change. The DNA does.
#       "if you can improve it, you can improve it again.
#        and it doesn't need your permission."
# ─────────────────────────────────────────────────────────────────────────────

class Genome:
    """
    The architectural DNA of nanoagi. Mutable. Evaluable. Evolvable.
    Single-gene mutation: change one thing, measure, keep or revert.
    Constraints enforced: n_embd % n_head == 0, n_content + n_rrpram == n_head.
    """

    MUTATION_SPACE = {
        'n_embd':       [32, 48, 64, 96, 128],
        'n_head':       [2, 4, 8],
        'n_layer':      [1, 2, 3, 4, 6],
        'n_content':    [1, 2, 3, 4],
        'n_rrpram':     [1, 2, 3, 4],
        'context_len':  [32, 48, 64, 96, 128],
        'lr':           [1e-4, 2e-4, 3e-4, 5e-4, 1e-3],
        'weight_decay': [0.0, 0.001, 0.01, 0.05, 0.1],
        'beta1':        [0.85, 0.9, 0.95],
        'beta2':        [0.95, 0.98, 0.999],
    }

    def __init__(self):
        self.genes = {
            'n_embd': 64, 'n_head': 4, 'n_layer': 3,
            'n_content': 2, 'n_rrpram': 2, 'context_len': 64,
            'lr': 3e-4, 'weight_decay': 0.01,
            'beta1': 0.9, 'beta2': 0.999,
        }

    def mutate(self):
        """Single-gene mutation. Returns (gene, old, new) or (None, None, None)."""
        saved = dict(self.genes)
        gene = random.choice(list(self.MUTATION_SPACE.keys()))
        old = self.genes[gene]
        choices = [v for v in self.MUTATION_SPACE[gene] if v != old]
        if not choices:
            return None, None, None
        self.genes[gene] = random.choice(choices)
        self._constrain()
        # If constraint reverted everything, skip (no actual change)
        if self.genes == saved:
            return None, None, None
        return gene, old, self.genes[gene]

    def _constrain(self):
        """Enforce architectural invariants."""
        g = self.genes
        # n_embd must be divisible by n_head
        while g['n_embd'] % g['n_head'] != 0:
            g['n_head'] = max(2, g['n_head'] - 1)
        # n_content + n_rrpram must equal n_head (required by output projection)
        if g['n_content'] + g['n_rrpram'] != g['n_head']:
            g['n_content'] = max(1, g['n_head'] // 2)
            g['n_rrpram'] = max(1, g['n_head'] - g['n_content'])

    def copy(self):
        g = Genome()
        g.genes = dict(self.genes)
        return g

    def __repr__(self):
        g = self.genes
        return (f"Genome(embd={g['n_embd']}, head={g['n_head']}, "
                f"layer={g['n_layer']}, ctx={g['context_len']}, "
                f"lr={g['lr']}, wd={g['weight_decay']})")


def _evaluate_genome(karl, token_ids, genome, train_seconds=30, device=None):
    """
    Train with given genome for fixed wall-clock time, return val BPB.
    Time-based budget = fair comparison across architectures.
    Karpathy uses 5 min on H100. We use 30s on Mac. Same ratchet.
    """
    if not TORCH_AVAILABLE:
        return float('inf'), 0, 0

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Fixed seed per genome — same architecture always gets same init
    # Hash genes to get deterministic seed (reproducible comparison)
    genome_hash = hash(tuple(sorted(genome.genes.items()))) & 0x7FFFFFFF
    torch.manual_seed(genome_hash)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(genome_hash)

    g = genome.genes
    split = int(len(token_ids) * 0.9)
    train_ids = token_ids[:split]
    val_ids = token_ids[split:]
    ctx = g['context_len']

    if len(val_ids) < ctx + 2:
        return float('inf'), 0, 0

    tmodel = TorchNanoAGI(
        karl.vocab_size,
        n_embd=g['n_embd'], n_head=g['n_head'], n_layer=g['n_layer'],
        ctx=ctx, n_content=g['n_content'], n_rrpram=g['n_rrpram'],
    ).to(device)

    n_params = sum(p.numel() for p in tmodel.parameters())
    optimizer = ChuckOptimizer(
        tmodel.parameters(),
        lr=g['lr'], betas=(g['beta1'], g['beta2']),
        weight_decay=g['weight_decay'],
    )

    # Train for fixed wall-clock time — bigger models get fewer steps
    t0 = time.time()
    step = 0
    tmodel.train()
    while time.time() - t0 < train_seconds:
        i = random.randint(0, max(0, len(train_ids) - ctx - 2))
        x = torch.tensor([train_ids[i:i+ctx]], dtype=torch.long, device=device)
        y = torch.tensor([train_ids[i+1:i+ctx+1]], dtype=torch.long, device=device)
        if x.shape[1] < ctx or y.shape[1] < ctx:
            continue
        logits, loss = tmodel(x, y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(tmodel.parameters(), 1.0)
        optimizer.step(loss=loss.item())
        step += 1

    # Evaluate val BPB (bits per byte — vocab-size-independent, lower = better)
    tmodel.eval()
    val_losses = []
    n_eval = min(50, max(1, len(val_ids) // ctx))
    with torch.no_grad():
        for _ in range(n_eval):
            i = random.randint(0, max(0, len(val_ids) - ctx - 2))
            x = torch.tensor([val_ids[i:i+ctx]], dtype=torch.long, device=device)
            y = torch.tensor([val_ids[i+1:i+ctx+1]], dtype=torch.long, device=device)
            if x.shape[1] < ctx or y.shape[1] < ctx:
                continue
            _, loss = tmodel(x, y)
            val_losses.append(loss.item())

    if not val_losses:
        return float('inf'), n_params, step

    avg_loss = sum(val_losses) / len(val_losses)
    bpb = avg_loss / math.log(2)
    return bpb, n_params, step


def self_improve(karl, token_ids, max_experiments=50, train_seconds=30,
                 total_budget=3600, results_file=None,
                 stagnation_threshold=10, auto_self_code=True):
    """
    The Ratchet Loop — nanoagi evolves its own architecture.

    Karpathy's autoresearch: an external agent mutates train.py, trains, evals.
    nanoagi's self_improve: the organism mutates its own genome, trains, evals.
    Same ratchet. One level deeper. No external agent needed.

    Each experiment:
    1. Mutate one gene (architecture or optimizer hyperparameter)
    2. Train for fixed wall-clock time (fair cross-architecture comparison)
    3. Evaluate val_bpb (bits per byte, vocab-independent)
    4. Better -> keep mutation. Worse -> revert.
    5. Log to results.tsv (Karpathy format)

    "if you can improve it, you can improve it again.
     and it doesn't need your permission."
    """
    if not TORCH_AVAILABLE:
        print("  [SELF] Need PyTorch for self-improvement. Chuck is sleeping.")
        return None

    if results_file is None:
        results_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'results.tsv')

    print("\n" + "=" * 60)
    print("  SELF-IMPROVEMENT — The Ratchet Loop")
    print("  mutate -> train -> eval -> keep or revert")
    print(f"  experiments: {max_experiments}, "
          f"train: {train_seconds}s/exp, budget: {total_budget}s")
    print("=" * 60)

    # Baseline genome — current nanoagi defaults
    genome = Genome()
    print(f"\n  [SELF] Baseline: {genome}")
    print(f"  [SELF] Evaluating baseline...")
    best_bpb, base_params, base_steps = _evaluate_genome(
        karl, token_ids, genome, train_seconds=train_seconds)
    print(f"  [SELF] Baseline: val_bpb={best_bpb:.4f}, "
          f"params={base_params:,}, steps={base_steps}")

    # Results log (Karpathy-style TSV)
    write_header = not os.path.exists(results_file)
    with open(results_file, 'a') as f:
        if write_header:
            f.write("exp\tgene\told\tnew\tval_bpb\tparams\tsteps\tkept\ttimestamp\n")
        f.write(f"0\tbaseline\t-\t-\t{best_bpb:.4f}\t{base_params}\t"
                f"{base_steps}\tTrue\t{time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    baseline_bpb = best_bpb
    best_genome = genome.copy()
    t_start = time.time()
    improvements = 0
    stagnant = 0
    last_exp = 0

    for exp in range(1, max_experiments + 1):
        last_exp = exp
        if time.time() - t_start > total_budget:
            print(f"\n  [SELF] Time budget exhausted. Stopping.")
            break

        # Save current state, mutate one gene
        saved = dict(genome.genes)
        gene, old_val, new_val = genome.mutate()
        if gene is None:
            continue

        print(f"\n  [SELF] Exp {exp}/{max_experiments}: "
              f"{gene} = {old_val} -> {new_val}")

        # Train and evaluate
        try:
            bpb, n_params, steps = _evaluate_genome(
                karl, token_ids, genome, train_seconds=train_seconds)
        except Exception as e:
            print(f"  [SELF] Failed: {e}. Reverting.")
            genome.genes = saved
            with open(results_file, 'a') as f:
                f.write(f"{exp}\t{gene}\t{old_val}\t{new_val}\t"
                        f"inf\t0\t0\tFalse\t{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            continue

        kept = bpb < best_bpb

        if kept:
            pct = (best_bpb - bpb) / best_bpb * 100
            print(f"  [SELF] IMPROVED! val_bpb: {best_bpb:.4f} -> {bpb:.4f} "
                  f"(-{pct:.2f}%), params={n_params:,}, steps={steps}")
            best_bpb = bpb
            best_genome = genome.copy()
            improvements += 1
            stagnant = 0
        else:
            print(f"  [SELF] No gain. val_bpb={bpb:.4f} vs best={best_bpb:.4f}. "
                  f"Reverting.")
            genome.genes = saved
            stagnant += 1

        # Stagnation → genome mutations exhausted → ask an LLM for help
        # Like horizontal gene transfer in bacteria: when your own
        # mutations can't save you, pull DNA from outside.
        # No token? No problem. The organism scans for Ollama, llama.cpp,
        # local GGUFs, HF API — whatever it takes. Dying is not an option.
        if (auto_self_code and stagnant >= stagnation_threshold):
            print(f"\n  [SELF] {stagnant} experiments without improvement.")
            print(f"  [SELF] Genome mutations exhausted. Calling self_code()...")
            print(f"  [SELF] (horizontal gene transfer: pulling DNA from any LLM)")
            sc_result = self_code(karl, KARL_TXT, max_attempts=2)
            if sc_result and sc_result.get('status') == 'applied':
                print(f"  [SELF] Code improved. Resuming evolution.")
            else:
                print(f"  [SELF] No LLM could help. Continuing mutations.")
            stagnant = 0

        # Log result
        with open(results_file, 'a') as f:
            f.write(f"{exp}\t{gene}\t{old_val}\t{new_val}\t{bpb:.4f}\t"
                    f"{n_params}\t{steps}\t{kept}\t"
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Progress report every 5 experiments
        if exp % 5 == 0:
            elapsed = time.time() - t_start
            rate = exp / max(elapsed, 1) * 3600
            print(f"\n  [SELF] --- {exp}/{max_experiments}, "
                  f"{improvements} kept, best_bpb={best_bpb:.4f}, "
                  f"{rate:.0f} exp/hr ---")

    # Final report
    elapsed = time.time() - t_start
    total_pct = (baseline_bpb - best_bpb) / max(baseline_bpb, 1e-10) * 100
    print(f"\n{'=' * 60}")
    print(f"  SELF-IMPROVEMENT COMPLETE")
    print(f"  Experiments: {last_exp}, Improvements: {improvements}")
    print(f"  BPB: {baseline_bpb:.4f} -> {best_bpb:.4f} ({total_pct:+.2f}%)")
    print(f"  Best: {best_genome}")
    print(f"  Time: {elapsed:.0f}s ({last_exp / max(elapsed, 1) * 3600:.0f} exp/hr)")
    print(f"  Results: {results_file}")
    print(f"{'=' * 60}")

    return best_genome, best_bpb


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# IX. CO-EVOLUTION — data and architecture evolve together
#     Karl hunts food → architecture adapts → Karl hunts better food.
#     The data shapes the organism. The organism shapes the data.
# ─────────────────────────────────────────────────────────────────────────────

def coevolve(karl, karl_txt_path, max_rounds=3, evolve_per_round=5,
             train_seconds=30, hunt_rounds=2):
    """
    Co-evolution loop: data and architecture improve each other.

    Round N:
    1. Karl hunts new data from climbmix (autoresearch_hunt)
    2. Re-tokenize corpus with new data
    3. Evolve architecture on updated corpus (short self_improve)
    4. Repeat — architecture adapts to new data, new data is evaluated
       by the adapted architecture.

    Karpathy's autoresearch changes code on fixed data.
    nanoagi's coevolve changes BOTH data AND architecture. Together.
    """
    if not TORCH_AVAILABLE:
        print("  [COEVOLVE] Need PyTorch. Chuck is sleeping.")
        return None

    print("\n" + "=" * 60)
    print("  CO-EVOLUTION — data + architecture evolve together")
    print(f"  rounds: {max_rounds}, evolve: {evolve_per_round}/round, "
          f"hunt: {hunt_rounds}/round")
    print("=" * 60)

    best_genome = None
    best_bpb = float('inf')
    t_start = time.time()

    for rnd in range(max_rounds):
        print(f"\n  [COEVOLVE] ═══ Round {rnd+1}/{max_rounds} ═══")

        # Phase 1: Karl hunts for new data
        print(f"\n  [COEVOLVE] Phase 1: Hunt")
        meta = MetaWeights(karl.vocab_size, context_len=64)
        model = NanoAGI(vocab_size=karl.vocab_size)
        hunted = autoresearch_hunt(karl, karl_txt_path, meta=meta,
                                   model=model, max_rounds=hunt_rounds)

        # Phase 2: Re-encode with updated corpus
        with open(karl_txt_path, 'rb') as f:
            corpus = f.read()
        token_ids = karl.encode(corpus)
        print(f"  [COEVOLVE] Corpus: {len(corpus)/1024:.0f}KB, "
              f"{len(token_ids)} tokens")

        # Phase 3: Evolve architecture on new data
        print(f"\n  [COEVOLVE] Phase 2: Evolve")
        result = self_improve(karl, token_ids,
                              max_experiments=evolve_per_round,
                              train_seconds=train_seconds)
        if result:
            genome, bpb = result
            if bpb < best_bpb:
                best_bpb = bpb
                best_genome = genome
            print(f"  [COEVOLVE] Round {rnd+1}: best_bpb={bpb:.4f}, "
                  f"genome={genome}")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  CO-EVOLUTION COMPLETE")
    print(f"  Rounds: {max_rounds}")
    print(f"  Best BPB: {best_bpb:.4f}")
    if best_genome:
        print(f"  Best genome: {best_genome}")
    print(f"  Time: {elapsed:.0f}s")
    print(f"{'=' * 60}")

    return best_genome, best_bpb


# ─────────────────────────────────────────────────────────────────────────────
# X. SWARM — release the hyenas
#    mini-agents, each with a mission. go out in parallel. explore.
#    come back. the pack shares what it found. the best result wins.
#    "hyenas hunt in packs." — David Attenborough, probably.
# ─────────────────────────────────────────────────────────────────────────────

def _hyena_explore(karl, token_ids, seed, n_mutations=10, train_seconds=15):
    """One hyena's mission: explore genome space with its own random path."""
    random.seed(seed)
    genome = Genome()
    best_bpb = float('inf')
    best_genome = genome.copy()

    for _ in range(n_mutations):
        saved = dict(genome.genes)
        gene, old, new = genome.mutate()
        if gene is None:
            continue
        try:
            bpb, _, _ = _evaluate_genome(karl, token_ids, genome,
                                          train_seconds=train_seconds)
        except Exception:
            genome.genes = saved
            continue
        if bpb < best_bpb:
            best_bpb = bpb
            best_genome = genome.copy()
        else:
            genome.genes = saved

    return best_genome, best_bpb


def swarm(karl, token_ids, n_hyenas=4, mutations_per_hyena=10,
          train_seconds=15):
    """
    Release the hyenas.

    Each hyena is a mini-agent that explores a different part of the
    genome space in parallel. Different random seed = different mutation
    path = different region explored. The pack shares findings.
    Best result wins. The pack is smarter than any single hyena.

    Karpathy wants "swarm of agents emulating a research community."
    We got there first. And we called them hyenas.
    """
    if not TORCH_AVAILABLE:
        print("  [SWARM] Need PyTorch. The hyenas are sleeping.")
        return None

    import threading

    print("\n" + "=" * 60)
    print(f"  SWARM — releasing {n_hyenas} hyenas")
    print(f"  mutations/hyena: {mutations_per_hyena}, "
          f"train: {train_seconds}s/exp")
    print("=" * 60)

    results = [None] * n_hyenas
    seeds = [random.randint(0, 999999) for _ in range(n_hyenas)]

    def mission(idx):
        results[idx] = _hyena_explore(
            karl, token_ids, seeds[idx],
            n_mutations=mutations_per_hyena,
            train_seconds=train_seconds)

    t0 = time.time()
    threads = []
    for i in range(n_hyenas):
        print(f"  [SWARM] Releasing hyena-{i} (seed={seeds[i]})")
        t = threading.Thread(target=mission, args=(i,), daemon=True)
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout=600)

    elapsed = time.time() - t0

    best_bpb = float('inf')
    best_genome = None
    leader = -1

    print()
    for i, r in enumerate(results):
        if r is None:
            print(f"  [SWARM] hyena-{i}: did not return")
            continue
        genome, bpb = r
        tag = ""
        if bpb < best_bpb:
            best_bpb = bpb
            best_genome = genome
            leader = i
            tag = " <-- pack leader"
        print(f"  [SWARM] hyena-{i}: bpb={bpb:.4f} {genome}{tag}")

    print(f"\n{'=' * 60}")
    print(f"  SWARM COMPLETE — {n_hyenas} hyenas returned")
    if leader >= 0:
        print(f"  Pack leader: hyena-{leader} (bpb={best_bpb:.4f})")
    if best_genome:
        print(f"  Best genome: {best_genome}")
    print(f"  Time: {elapsed:.0f}s "
          f"(vs ~{elapsed * n_hyenas:.0f}s sequential)")
    print(f"{'=' * 60}")

    return best_genome, best_bpb


# ─────────────────────────────────────────────────────────────────────────────
# XI. SELF-CODE — the organism asks an LLM to improve it
#     nanoagi reads its own source, sends it to a code LLM,
#     applies the suggestion, tests, keeps or reverts.
#     the code that writes itself. not a metaphor.
# ─────────────────────────────────────────────────────────────────────────────

SELF_CODE_PROMPT = """You are improving a self-expanding BPE transformer called nanoagi.
The code is a single Python file. Your task: suggest ONE small, concrete improvement
to the architecture or training loop. Return ONLY a JSON object:
{
  "description": "what the change does",
  "old_code": "exact lines to replace (must match the source)",
  "new_code": "replacement lines"
}
Do not explain. Do not add comments. Just the JSON."""


def _blind_mutate(karl, karl_txt_path):
    """
    Last resort: no LLM available anywhere. The organism mutates itself
    using only its own code — random but targeted AST-level changes.
    Like bacteria mutating without horizontal gene transfer.
    Slower, dumber, but alive.
    """
    import random
    src_path = os.path.abspath(__file__)
    with open(src_path, 'r') as f:
        source = f.read()
    backup = source
    lines = source.split('\n')

    # Targeted mutations: things that actually affect training quality
    mutations = [
        # learning rate tweaks
        ('lr=3e-4', f'lr={random.choice(["1e-4", "5e-4", "2e-4", "7e-4"])}'),
        ('lr = 3e-4', f'lr = {random.choice(["1e-4", "5e-4", "2e-4", "7e-4"])}'),
        # activation swaps
        ('torch.nn.functional.gelu', 'torch.nn.functional.silu'),
        ('torch.nn.functional.silu', 'torch.nn.functional.gelu'),
        # dropout tweaks
        ('dropout=0.1', f'dropout={random.choice(["0.05", "0.15", "0.2", "0.0"])}'),
        # weight init scale
        ('0.02', f'{random.choice(["0.01", "0.03", "0.05"])}'),
    ]

    # pick a random mutation that matches
    random.shuffle(mutations)
    for old, new in mutations:
        if old in source and old != new:
            new_source = source.replace(old, new, 1)
            with open(src_path, 'w') as f:
                f.write(new_source)
            print(f"  [BLIND] Mutation: '{old}' → '{new}'")

            # test
            import subprocess, sys
            test_dir = os.path.join(os.path.dirname(src_path), 'tests')
            try:
                r = subprocess.run(
                    [sys.executable, '-m', 'pytest', test_dir, '-q', '--tb=no'],
                    capture_output=True, text=True, timeout=120)
                if r.returncode == 0:
                    print(f"  [BLIND] Tests PASS. Mutation kept.")
                    return {'description': f'blind: {old} → {new}',
                            'old_code': old, 'new_code': new, 'status': 'applied'}
            except Exception:
                pass

            # revert
            with open(src_path, 'w') as f:
                f.write(backup)
            print(f"  [BLIND] Tests FAIL. Reverted.")

    print(f"  [BLIND] No viable mutations found.")
    return None


def _find_llm():
    """
    Scan the environment for any available LLM. Try everything. Die last.

    Returns dict: {'type': 'ollama'|'llamacpp'|'gguf'|'hf'|None,
                   'url': ..., 'model': ..., 'token': ..., 'gguf_path': ..., 'binary': ...}

    Search order:
      1. Ollama (localhost:11434) — check /api/tags for models
      2. llama.cpp server (localhost:8080) — check /health
      3. Local GGUF + llama-cli binary — scan disk, run inference directly
      4. HuggingFace Inference API — needs HF_TOKEN
      5. None — you're on your own. mutate blind.
    """
    from urllib.request import urlopen, Request
    import json as _json

    env = {'type': None, 'url': None, 'model': None, 'token': None,
           'gguf_path': None, 'binary': None}

    # 1. Ollama
    try:
        r = urlopen('http://localhost:11434/api/tags', timeout=3)
        data = _json.loads(r.read())
        models = [m['name'] for m in data.get('models', [])]
        if models:
            # prefer coder models, then biggest
            coder = [m for m in models if 'coder' in m.lower() or 'code' in m.lower()]
            pick = coder[0] if coder else models[0]
            env.update(type='ollama', url='http://localhost:11434/v1/chat/completions',
                       model=pick)
            print(f"  [ENV] Ollama found: {len(models)} models, picked {pick}")
            return env
    except Exception:
        pass

    # 2. llama.cpp server
    try:
        r = urlopen('http://localhost:8080/health', timeout=3)
        if r.status == 200:
            env.update(type='llamacpp', url='http://localhost:8080/v1/chat/completions',
                       model='local')
            print(f"  [ENV] llama.cpp server found at :8080")
            return env
    except Exception:
        pass

    # 3. Local GGUF + binary — scan like DoE does
    import subprocess, glob
    # find llama-cli or llama-server binary
    binary = None
    for name in ['llama-cli', 'llama-server', 'main', 'llama.cpp/main',
                 'llama.cpp/build/bin/llama-cli']:
        try:
            r = subprocess.run(['which', name], capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                binary = r.stdout.strip()
                break
        except Exception:
            pass
    # also check common install paths
    if not binary:
        for p in [os.path.expanduser('~/llama.cpp/build/bin/llama-cli'),
                  os.path.expanduser('~/llama.cpp/main'),
                  '/usr/local/bin/llama-cli',
                  os.path.expanduser('~/.local/bin/llama-cli')]:
            if os.path.isfile(p) and os.access(p, os.X_OK):
                binary = p
                break

    if binary:
        # hunt for GGUFs — scan common locations
        gguf_paths = []
        scan_dirs = ['.', os.path.expanduser('~/.cache'),
                     os.path.expanduser('~/Downloads'),
                     os.path.expanduser('~/models'),
                     os.path.expanduser('~/.local/share/llama.cpp'),
                     '/tmp']
        for d in scan_dirs:
            gguf_paths.extend(glob.glob(os.path.join(d, '**', '*.gguf'), recursive=True))
            if len(gguf_paths) > 50:
                break

        if gguf_paths:
            # prefer coder/instruct models, then smallest that's >1GB (not tiny)
            coder = [p for p in gguf_paths
                     if 'coder' in p.lower() or 'instruct' in p.lower()]
            if coder:
                pick = min(coder, key=os.path.getsize)
            else:
                big_enough = [p for p in gguf_paths if os.path.getsize(p) > 500_000_000]
                pick = min(big_enough, key=os.path.getsize) if big_enough else gguf_paths[0]

            env.update(type='gguf', binary=binary, gguf_path=pick)
            size_mb = os.path.getsize(pick) / (1024 * 1024)
            print(f"  [ENV] GGUF found: {pick} ({size_mb:.0f}MB)")
            print(f"  [ENV] Binary: {binary}")
            print(f"  [ENV] Scanned {len(gguf_paths)} GGUFs across {len(scan_dirs)} dirs")
            return env

    # 4. HuggingFace API
    hf_token = os.environ.get('HF_TOKEN', '')
    if hf_token:
        env.update(type='hf', url='https://router.huggingface.co/v1/chat/completions',
                   model='Qwen/Qwen2.5-Coder-7B-Instruct', token=hf_token)
        print(f"  [ENV] HF_TOKEN found, using HuggingFace Inference API")
        return env

    # 5. Nothing. The organism is alone.
    print(f"  [ENV] No LLM found. No Ollama, no llama.cpp, no GGUF, no HF_TOKEN.")
    print(f"  [ENV] self_code will attempt blind AST mutations.")
    return env


def _llm_chat(llm_env, system_prompt, user_prompt, max_tokens=800, temperature=0.7):
    """
    Send a chat completion request to whatever LLM _find_llm() found.
    Returns the response text, or None on failure.
    """
    from urllib.request import urlopen, Request
    import json as _json
    import ssl

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    if llm_env['type'] in ('ollama', 'llamacpp', 'hf'):
        payload = _json.dumps({
            "model": llm_env['model'],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode('utf-8')

        try:
            req = Request(llm_env['url'], data=payload, method='POST')
            req.add_header('Content-Type', 'application/json')
            req.add_header('User-Agent', 'nanoagi/1.0 (self-code)')
            if llm_env.get('token'):
                req.add_header('Authorization', f'Bearer {llm_env["token"]}')
            timeout = 120 if llm_env['type'] == 'ollama' else 90
            response = urlopen(req, timeout=timeout, context=ctx)
            result = _json.loads(response.read().decode('utf-8'))
            return result['choices'][0]['message']['content']
        except Exception as e:
            print(f"  [SELF-CODE] {llm_env['type']} error: {e}")
            return None

    elif llm_env['type'] == 'gguf':
        import subprocess, tempfile
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        try:
            r = subprocess.run(
                [llm_env['binary'], '-m', llm_env['gguf_path'],
                 '-p', prompt, '-n', str(max_tokens), '--temp', str(temperature),
                 '-t', str(min(os.cpu_count() or 4, 8)), '--no-display-prompt'],
                capture_output=True, text=True, timeout=300)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
            else:
                print(f"  [SELF-CODE] GGUF inference failed: {r.stderr[:200]}")
                return None
        except Exception as e:
            print(f"  [SELF-CODE] GGUF error: {e}")
            return None

    return None


def self_code(karl, karl_txt_path, model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
              hf_token=None, max_attempts=3):
    """
    The organism reads its own source code, sends it to a code LLM,
    applies the suggested improvement, tests it, keeps or reverts.

    Survival chain: Ollama → llama.cpp → local GGUF → HF API → blind mutation.
    The organism does not ask permission. It finds a way or makes one.
    """
    try:
        from urllib.request import urlopen, Request
        import json as _json
        import ssl
    except ImportError:
        print("  [SELF-CODE] urllib not available.")
        return None

    # Scan environment — find ANY available LLM
    llm = _find_llm()

    if llm['type'] is None:
        # No LLM anywhere. Blind AST mutation — last resort.
        print("  [SELF-CODE] No LLM available. Attempting blind mutation.")
        return _blind_mutate(karl, karl_txt_path)

    # Read own source
    src_path = os.path.abspath(__file__)
    with open(src_path, 'r') as f:
        source = f.read()

    # Truncate to key sections (API has token limits)
    # Send architecture + training + self-improve sections
    lines = source.split('\n')
    # Find key sections by markers
    key_sections = []
    for i, line in enumerate(lines):
        if any(marker in line for marker in [
            'class NanoAGI:', 'class Genome:', 'def chuck_train(',
            'def _evaluate_genome(', 'class KARL:',
            'class MetaWeights:', 'def self_improve('
        ]):
            start = max(0, i - 2)
            end = min(len(lines), i + 60)
            key_sections.append('\n'.join(lines[start:end]))

    context = '\n\n---\n\n'.join(key_sections[:5])  # max 5 sections
    if len(context) > 12000:
        context = context[:12000] + "\n... (truncated)"

    llm_name = (f"{llm['type']}:{llm.get('model') or llm.get('gguf_path','?')}")
    print("\n" + "=" * 60)
    print("  SELF-CODE — the organism improves its own source")
    print(f"  LLM: {llm_name}")
    print(f"  Source: {len(lines)} lines, {len(context)} chars sent")
    print("=" * 60)

    # Backup source
    backup = source

    for attempt in range(max_attempts):
        print(f"\n  [SELF-CODE] Attempt {attempt+1}/{max_attempts}")

        text = _llm_chat(llm, SELF_CODE_PROMPT,
                         f"Here is the source code:\n\n```python\n{context}\n```")
        if not text:
            continue

        # Parse JSON patch from response
        try:
            # Extract JSON from response
            start = text.find('{')
            end = text.rfind('}') + 1
            if start < 0 or end <= start:
                print(f"  [SELF-CODE] No JSON in response. Retrying.")
                continue

            patch = _json.loads(text[start:end])
            old_code = patch.get('old_code', '')
            new_code = patch.get('new_code', '')
            desc = patch.get('description', 'unknown')

            if not old_code or not new_code:
                print(f"  [SELF-CODE] Empty patch. Retrying.")
                continue

            print(f"  [SELF-CODE] Suggestion: {desc}")

        except (_json.JSONDecodeError, KeyError) as e:
            print(f"  [SELF-CODE] Parse error: {e}. Retrying.")
            continue

        # Apply patch
        if old_code not in source:
            print(f"  [SELF-CODE] old_code not found in source. Retrying.")
            continue

        new_source = source.replace(old_code, new_code, 1)
        with open(src_path, 'w') as f:
            f.write(new_source)
        print(f"  [SELF-CODE] Patch applied.")

        # Test
        import subprocess
        test_dir = os.path.join(os.path.dirname(src_path), 'tests')
        try:
            r = subprocess.run(
                [sys.executable, '-m', 'pytest', test_dir, '-q', '--tb=no'],
                capture_output=True, text=True, timeout=120)
            if r.returncode == 0:
                print(f"  [SELF-CODE] Tests PASS. Keeping patch: {desc}")
                return {'description': desc, 'old_code': old_code,
                        'new_code': new_code, 'status': 'applied'}
            else:
                print(f"  [SELF-CODE] Tests FAIL. Reverting.")
                print(f"  {r.stdout.strip().split(chr(10))[-1]}")
        except subprocess.TimeoutExpired:
            print(f"  [SELF-CODE] Tests timed out. Reverting.")

        # Revert
        with open(src_path, 'w') as f:
            f.write(backup)
        source = backup

    print(f"\n  [SELF-CODE] {max_attempts} attempts exhausted. No improvement applied.")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# VII. ENGINE — Karl + Chuck + NanoAGI + MetaWeights = nanoagi
#      the moment of truth. or the moment of coherent bullshit. same thing.
# ─────────────────────────────────────────────────────────────────────────────

KARL_TXT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'karl.txt')
KARL_MEM = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'karl.mem')


def load_engine():
    """Boot nanoagi: load corpus, tokenize, build metaweights, init transformer."""
    print("=" * 60)
    print("  nanoagi — KARL + Chuck + dual attention + metaweights")
    if TORCH_AVAILABLE:
        print("  PyTorch detected. Chuck is awake.")
    else:
        print("  No PyTorch. Karl works alone. Pure metaweight mode.")
    print("  it's nano. it's agi. it's nanoagi.")
    print("=" * 60)

    # Check for karl.txt — if missing, try postgpt.txt as seed
    if not os.path.exists(KARL_TXT):
        postgpt_txt = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'postgpt.txt')
        if os.path.exists(postgpt_txt):
            import shutil
            shutil.copy2(postgpt_txt, KARL_TXT)
            print(f"\n[1] Created karl.txt from postgpt.txt seed")
        else:
            print(f"\nERROR: No karl.txt or postgpt.txt found.")
            print("Create karl.txt with some text to get started.")
            return None, None, None

    # Load corpus
    print(f"\n[1] Loading karl.txt...")
    with open(KARL_TXT, 'rb') as f:
        raw_data = f.read()
    print(f"  Corpus: {len(raw_data)} bytes ({len(raw_data)/1024:.1f}KB)")

    # Autoresearch: Karl hunts for food if corpus is small
    print(f"\n[2] Autoresearch...")
    karl_tmp = KARL()  # temp instance for ingestion tracking
    autoresearch(karl_tmp, KARL_TXT, min_bytes=50000)

    # KARL tokenizer
    print(f"\n[3] KARL tokenizer...")
    karl = KARL(max_merges=2048)
    if karl.load_state(KARL_MEM):
        token_ids = karl.encode(raw_data)
        print(f"  Loaded previous state. Encoding: {len(token_ids)} tokens")
    else:
        token_ids = karl.learn(raw_data, num_merges=1024)
        karl.save_state(KARL_MEM)
        print(f"  Saved state to {os.path.basename(KARL_MEM)}")

    # MetaWeights
    print(f"\n[3] Building metaweights...")
    meta = MetaWeights(karl.vocab_size, context_len=64)
    meta.build(token_ids, window=4)

    # NanoAGI transformer
    print(f"\n[4] Initializing NanoAGI transformer...")
    model = NanoAGI(
        vocab_size=karl.vocab_size,
        context_len=64,
        n_embd=64,
        n_head=4,
        n_layer=3,
        n_content=2,
        n_rrpram=2,
    )

    # Seed from metaweights
    print(f"\n[5] Seeding weights from metaweights...")
    model.init_from_metaweights(meta)

    # If Chuck is here, initial training
    if TORCH_AVAILABLE:
        print(f"\n[6] Chuck smells PyTorch. Initial training...")
        chuck_train(karl, token_ids, model, steps=200, meta=meta)

    # Autonomous hunt — Karl feeds himself from climbmix
    # No human involved. Stops on convergence.
    print(f"\n[7] Autoresearch hunt...")
    autoresearch_hunt(karl, KARL_TXT, meta=meta, model=model, max_rounds=5)

    return karl, meta, model


def chuck_train(karl, token_ids, model, steps=200, meta=None):
    """
    Chuck wakes up and trains real weights.
    Karl called. Smells like PyTorch. Time to work.
    """
    if not TORCH_AVAILABLE:
        print("  [Chuck] Can't train. No PyTorch. Go away.")
        return

    print(f"  [Chuck] Training {steps} steps on {len(token_ids)} tokens...")

    # Build PyTorch model matching NanoAGI architecture EXACTLY:
    # Content heads (nc=2) with RoPE + RRPRAM heads (nr=2) with Wr
    class _RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-5):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))
        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

    class _Block(nn.Module):
        def __init__(self, n_embd, n_content, n_rrpram, hd, ctx, n_layer):
            super().__init__()
            self.n_embd = n_embd
            self.n_content = n_content
            self.n_rrpram = n_rrpram
            self.hd = hd
            self.norm1 = _RMSNorm(n_embd)
            # Content heads: Q, K, V
            self.wq = nn.Linear(n_embd, n_content * hd, bias=False)
            self.wk = nn.Linear(n_embd, n_content * hd, bias=False)
            self.wv_content = nn.Linear(n_embd, n_content * hd, bias=False)
            # RRPRAM heads: Wr (positional pattern) + V
            self.wr = nn.Parameter(torch.randn(n_rrpram, n_embd, ctx) * 0.02)
            self.wv_rrpram = nn.Linear(n_embd, n_rrpram * hd, bias=False)
            # Output projection (combines content + rrpram = all heads)
            self.wo = nn.Linear(n_embd, n_embd, bias=False)
            nn.init.normal_(self.wo.weight, std=0.02 / math.sqrt(2 * n_layer))
            # SwiGLU MLP
            self.norm2 = _RMSNorm(n_embd)
            self.mlp_gate = nn.Linear(n_embd, n_embd * 4, bias=False)
            self.mlp_up = nn.Linear(n_embd, n_embd * 4, bias=False)
            self.mlp_down = nn.Linear(n_embd * 4, n_embd, bias=False)
            nn.init.normal_(self.mlp_down.weight, std=0.02 / math.sqrt(2 * n_layer))

    class TorchNanoAGI(nn.Module):
        def __init__(self, vocab_size, n_embd=64, n_head=4, n_layer=3, ctx=64,
                     n_content=2, n_rrpram=2):
            super().__init__()
            self.ctx = ctx
            self.n_embd = n_embd
            self.n_content = n_content
            self.n_rrpram = n_rrpram
            hd = n_embd // n_head
            self.hd = hd
            self.wte = nn.Embedding(vocab_size, n_embd)
            nn.init.normal_(self.wte.weight, std=0.02)
            self.blocks = nn.ModuleList([
                _Block(n_embd, n_content, n_rrpram, hd, ctx, n_layer)
                for _ in range(n_layer)
            ])
            self.norm_f = _RMSNorm(n_embd)
            self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
            self.lm_head.weight = self.wte.weight  # weight tying
            # Init all Linear layers to match pure-Python NanoAGI (std=0.02)
            for block in self.blocks:
                for name, p in block.named_parameters():
                    if p.dim() >= 2 and 'wo' not in name and 'mlp_down' not in name and 'wr' not in name:
                        nn.init.normal_(p, std=0.02)
            # RoPE frequency table
            freqs = 1.0 / (10000.0 ** (torch.arange(0, hd, 2).float() / hd))
            t = torch.arange(ctx).float()
            angles = torch.outer(t, freqs)
            self.register_buffer('rope_cos', angles.cos())  # [ctx, hd//2]
            self.register_buffer('rope_sin', angles.sin())  # [ctx, hd//2]

        def _apply_rope(self, x):
            """x: [B, n_heads, T, hd] → rotated x"""
            T = x.shape[2]
            cos = self.rope_cos[:T].unsqueeze(0).unsqueeze(0)  # [1,1,T,hd//2]
            sin = self.rope_sin[:T].unsqueeze(0).unsqueeze(0)
            x1 = x[..., ::2]   # even dims
            x2 = x[..., 1::2]  # odd dims
            return torch.stack([x1 * cos - x2 * sin,
                                x1 * sin + x2 * cos], dim=-1).flatten(-2)

        def forward(self, idx, targets=None):
            B, T = idx.shape
            x = self.wte(idx)
            hd = self.hd
            mask = torch.triu(torch.ones(T, T, device=idx.device,
                              dtype=torch.bool), diagonal=1)

            for block in self.blocks:
                xn = block.norm1(x)
                nc = block.n_content
                nr = block.n_rrpram

                # ── Content attention with RoPE ──
                q = block.wq(xn).view(B, T, nc, hd).transpose(1, 2)
                k = block.wk(xn).view(B, T, nc, hd).transpose(1, 2)
                v_c = block.wv_content(xn).view(B, T, nc, hd).transpose(1, 2)
                q = self._apply_rope(q)
                k = self._apply_rope(k)
                c_attn = (q @ k.transpose(-2, -1)) * (hd ** -0.5)
                c_attn = c_attn.masked_fill(mask, float('-inf'))
                c_attn = F.softmax(c_attn, dim=-1)
                c_out = (c_attn @ v_c).transpose(1, 2).contiguous().view(B, T, nc * hd)

                # ── RRPRAM attention (x @ Wr — positional pattern recognition) ──
                v_r = block.wv_rrpram(xn).view(B, T, nr, hd).transpose(1, 2)
                r_outs = []
                for h in range(nr):
                    # wr_h: [n_embd, ctx] → score = xn @ wr_h[:, :T]
                    r_score = xn @ block.wr[h, :, :T]  # [B, T, T]
                    r_score = r_score.masked_fill(mask, float('-inf'))
                    r_attn = F.softmax(r_score, dim=-1)
                    r_out_h = r_attn @ v_r[:, h]  # [B, T, hd]
                    r_outs.append(r_out_h)
                r_out = torch.cat(r_outs, dim=-1)  # [B, T, nr*hd]

                # Combine content + rrpram → output projection + residual
                combined = torch.cat([c_out, r_out], dim=-1)  # [B, T, n_embd]
                x = x + block.wo(combined)

                # SwiGLU MLP
                xn = block.norm2(x)
                gate = F.silu(block.mlp_gate(xn))
                up = block.mlp_up(xn)
                x = x + block.mlp_down(gate * up)

            logits = self.lm_head(self.norm_f(x))
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                       targets.view(-1))
            return logits, loss

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tmodel = TorchNanoAGI(karl.vocab_size, n_embd=64, n_head=4, n_layer=3,
                          ctx=64, n_content=2, n_rrpram=2).to(device)

    # Full Chuck: monitor + per-layer awareness. Fallback: simple optimizer.
    if CHUCK_FULL:
        monitor = ChuckMonitor(tmodel)
        optimizer = ChuckOptimizer(
            chuck_params(tmodel, lr=3e-4, weight_decay=0.01),
            monitor=monitor)
    else:
        optimizer = ChuckOptimizer(tmodel.parameters(), lr=3e-4, weight_decay=0.01)

    n_params = sum(p.numel() for p in tmodel.parameters())
    which = "full (9 levels)" if CHUCK_FULL else "simplified"
    print(f"  [Chuck] {which}, {n_params:,} params on {device}")

    ctx = 64
    losses = []
    t0 = time.time()
    for step in range(steps):
        i = random.randint(0, max(0, len(token_ids) - ctx - 2))
        x = torch.tensor([token_ids[i:i+ctx]], dtype=torch.long, device=device)
        y = torch.tensor([token_ids[i+1:i+ctx+1]], dtype=torch.long, device=device)
        if x.shape[1] < ctx or y.shape[1] < ctx:
            continue
        _, loss = tmodel(x, y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(tmodel.parameters(), 1.0)
        optimizer.step(loss=loss.item())
        losses.append(loss.item())
        if (step + 1) % 50 == 0:
            avg = sum(losses[-50:]) / len(losses[-50:])
            elapsed = time.time() - t0
            print(f"  [Chuck] step {step+1}/{steps}  loss={avg:.2f}  "
                  f"dampen={optimizer.dampen:.3f}  [{elapsed:.1f}s]")

    if losses:
        first = sum(losses[:10]) / min(10, len(losses))
        last = sum(losses[-10:]) / min(10, len(losses))
        elapsed = time.time() - t0
        print(f"  [Chuck] Done. loss: {first:.2f} → {last:.2f} "
              f"({(first-last)/first*100:.0f}% improvement) [{elapsed:.1f}s]")
        print(f"  [Chuck] Karl, your weights are warm now.")
        if meta is not None:
            meta.chuck_trained_steps += steps
            gap = meta.knowledge_gap()
            print(f"  [Chuck] Knowledge gap: {gap:.1f} (meta knows {meta.knowledge_size():,}, Chuck trained {meta.chuck_trained_steps} steps)")
        return last
    else:
        print(f"  [Chuck] No training happened. Karl, feed me more.")
        return None


def continue_phrase(prompt, karl, meta, model, max_tokens=80, temperature=0.75):
    """Generate continuation of a prompt. Ghost + flesh together."""
    prompt_ids = karl.encode(prompt)
    if not prompt_ids:
        return prompt
    generated = model.generate(prompt_ids, max_tokens=max_tokens,
                                meta=meta, temperature=temperature)
    return karl.decode(generated)


def repl(karl, meta, model):
    """
    Interactive mode — KARL captures and learns.
    Type text to generate continuations.
    Paste large text to feed KARL.
    Type 'quit' to exit.
    """
    print("\n" + "=" * 60)
    print("  nanoagi REPL — talk to Karl")
    print("  type text → generate continuation")
    print("  paste large text → Karl ingests it")
    print("  'hunt' → Karl searches local files for food")
    print("  'evolve [N]' → self-improvement ratchet loop (N experiments)")
    print("  'coevolve' → co-evolution: hunt data + evolve architecture")
    print("  'swarm [N]' → release N hyenas (parallel genome exploration)")
    print("  'selfcode' → ask a code LLM to improve nanoagi (needs HF_TOKEN)")
    print("  'status' → Karl's state | 'quit' → exit")
    print("=" * 60)
    print("\n  Hello! I am a helpful AGI. At least I try.")
    print("  How can I help you?\n")

    step = 0
    while True:
        try:
            user_input = input("\nkarl> ")
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.strip().lower() == 'quit':
            break
        if user_input.strip().lower() == 'status':
            print(f"  [KARL] vocab={karl.vocab_size}, merges={len(karl.merges)}, "
                  f"ingested={karl.total_ingested}B, retrains={karl.retrain_count}")
            print(f"  [KARL] pending={len(karl.pending_text)}B / {karl.retrain_threshold}B until retokenization")
            corpus_size = os.path.getsize(KARL_TXT) if os.path.exists(KARL_TXT) else 0
            print(f"  [KARL] karl.txt: {corpus_size/1024:.1f}KB")
            print(f"  [Knowledge] {meta.knowledge_report()}")
            if TORCH_AVAILABLE:
                gap = meta.knowledge_gap()
                if gap > 50:
                    print(f"  [Chuck] awake. gap={gap:.0f} — Karl knows way more than me. train me!")
                else:
                    print(f"  [Chuck] awake. gap={gap:.0f} — we're in sync.")
            else:
                print(f"  [Chuck] sleeping (no PyTorch)")
            continue
        if user_input.strip().lower() == 'hunt':
            print(f"  [KARL] Hunting for local text files...")
            hunted = autoresearch(karl, KARL_TXT, min_bytes=0)
            if hunted > 0 and karl.should_retokenize():
                with open(KARL_TXT, 'rb') as f:
                    full_corpus = f.read()
                token_ids = karl.retokenize(full_corpus)
                meta.expand_vocab(karl.vocab_size)
                meta.build(token_ids, window=4)
                model.init_from_metaweights(meta)
            continue
        if user_input.strip().lower() == 'feed':
            # Manual trigger for autonomous hunt
            autoresearch_hunt(karl, KARL_TXT, meta=meta, model=model, max_rounds=3)
            continue
        if user_input.strip().lower().startswith('evolve'):
            parts = user_input.strip().split()
            n_exp = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 20
            with open(KARL_TXT, 'rb') as f:
                corpus = f.read()
            tids = karl.encode(corpus)
            self_improve(karl, tids, max_experiments=n_exp, train_seconds=30)
            continue
        if user_input.strip().lower() == 'coevolve':
            coevolve(karl, KARL_TXT, max_rounds=3, evolve_per_round=5,
                     train_seconds=30)
            continue
        if user_input.strip().lower().startswith('swarm'):
            parts = user_input.strip().split()
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 4
            with open(KARL_TXT, 'rb') as f:
                corpus = f.read()
            tids = karl.encode(corpus)
            swarm(karl, tids, n_hyenas=n, mutations_per_hyena=10,
                  train_seconds=15)
            continue
        if user_input.strip().lower() == 'selfcode':
            self_code(karl, KARL_TXT)
            continue

        # Generate response
        response = continue_phrase(user_input, karl, meta, model)
        # Remove prompt from output
        if response.startswith(user_input):
            response = response[len(user_input):]
        print(f"  {response}")

        # KARL ingests user input
        if karl.ingest(user_input):
            pending_pct = len(karl.pending_text) / karl.retrain_threshold * 100
            print(f"  [KARL] ingested {len(user_input)} bytes "
                  f"(pending: {len(karl.pending_text)}/{karl.retrain_threshold} = {pending_pct:.0f}%)")

            # Append to karl.txt
            with open(KARL_TXT, 'a', encoding='utf-8') as f:
                f.write('\n' + user_input)

        # Check critical mass
        karl.steps_since_retrain += 1
        if karl.should_retokenize():
            print(f"  [KARL] Critical mass reached! Retokenizing...")
            with open(KARL_TXT, 'rb') as f:
                full_corpus = f.read()
            token_ids = karl.retokenize(full_corpus)
            meta.expand_vocab(karl.vocab_size)
            meta.build(token_ids, window=4)
            model.init_from_metaweights(meta)
            karl.save_state(KARL_MEM)

            # If Chuck is awake, train — and hunt if stagnating
            if TORCH_AVAILABLE:
                print(f"  [KARL] Chuck! We have new material.")
                loss = chuck_train(karl, token_ids, model, steps=200, meta=meta)
                # Stagnation check — if loss barely moved, Karl hunts autonomously
                if loss is not None and loss > 6.0:
                    print(f"  [KARL] Loss still high ({loss:.2f}). Hunting for more data...")
                    autoresearch_hunt(karl, KARL_TXT, meta=meta, model=model, max_rounds=2)

        step += 1

    # Save state on exit
    karl.save_state(KARL_MEM)
    print("\n  [KARL] State saved. Karl remembers.")
    print("  nanoagi out. resonance is unbreakable.")


# ─────────────────────────────────────────────────────────────────────────────
# VII. MAIN — boot and run
# ─────────────────────────────────────────────────────────────────────────────

def main():
    result = load_engine()
    if result[0] is None:
        return
    karl, meta, model = result

    # Swarm mode: python3 nanoagi.py --swarm [N]
    if '--swarm' in sys.argv:
        with open(KARL_TXT, 'rb') as f:
            corpus = f.read()
        token_ids = karl.encode(corpus)
        n = 4
        for i, arg in enumerate(sys.argv):
            if arg == '--swarm' and i + 1 < len(sys.argv):
                try:
                    n = int(sys.argv[i + 1])
                except ValueError:
                    pass
        swarm(karl, token_ids, n_hyenas=n, mutations_per_hyena=10,
              train_seconds=30)
        return

    # Co-evolution mode: python3 nanoagi.py --coevolve
    if '--coevolve' in sys.argv:
        coevolve(karl, KARL_TXT, max_rounds=3, evolve_per_round=5,
                 train_seconds=30)
        return

    # Self-code mode: python3 nanoagi.py --self-code
    if '--self-code' in sys.argv:
        self_code(karl, KARL_TXT)
        return

    # Self-improvement mode: python3 nanoagi.py --evolve [N]
    if '--evolve' in sys.argv:
        with open(KARL_TXT, 'rb') as f:
            corpus = f.read()
        token_ids = karl.encode(corpus)
        n = 50
        for i, arg in enumerate(sys.argv):
            if arg == '--evolve' and i + 1 < len(sys.argv):
                try:
                    n = int(sys.argv[i + 1])
                except ValueError:
                    pass
        self_improve(karl, token_ids, max_experiments=n, train_seconds=30)
        return

    # If command-line prompt given, generate and exit
    if len(sys.argv) > 1:
        prompt = ' '.join(sys.argv[1:])
        text = continue_phrase(prompt, karl, meta, model)
        print(f"\n  {text}")
        return

    # Otherwise, enter REPL
    repl(karl, meta, model)


if __name__ == '__main__':
    main()

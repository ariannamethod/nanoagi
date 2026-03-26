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
it's not AGI. it just doesn't know that yet.
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
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
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
    if you can improve it, it's not AGI yet.
    but it's getting closer.
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
        """Pure metaweight generation. No transformer. Just the ghost."""
        if meta is None:
            return prompt_ids
        if temperature is None:
            temperature = self.temperature
        generated = list(prompt_ids)
        recent_tokens = {}
        for _ in range(max_tokens):
            ctx = generated[-8:]
            last = generated[-1]
            bigram = meta.query_bigram(last, self.vocab_size)
            trigram = meta.query_trigram(generated[-2], last, self.vocab_size) if len(generated) >= 2 else [0.0] * self.vocab_size
            hebbian = meta.query_hebbian(ctx, self.vocab_size)
            prophecy = meta.query_prophecy(ctx, self.vocab_size)
            # Combine
            scores = [0.0] * self.vocab_size
            for i in range(self.vocab_size):
                scores[i] = (12.0 * bigram[i] + 8.0 * trigram[i] +
                             0.5 * hebbian[i] + 0.3 * prophecy[i] +
                             0.01 * meta.unigram[i])
                # Repetition penalty
                if i in recent_tokens:
                    scores[i] *= 1.0 / (1.0 + 0.5 * recent_tokens[i])
            # Top-k + temperature
            indexed = sorted(enumerate(scores), key=lambda x: -x[1])[:15]
            tokens_k = [t for t, _ in indexed]
            counts_k = [s for _, s in indexed]
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
            recent_tokens[chosen] = recent_tokens.get(chosen, 0) + 1
        return generated


# ─────────────────────────────────────────────────────────────────────────────
# V. CHUCK OPTIMIZER — self-aware learning. appears when PyTorch is around.
#    Chuck wakes up when he smells gradients.
#    Karl calls Chuck when there's enough new food.
#    together they are nanoagi.
# ─────────────────────────────────────────────────────────────────────────────

if TORCH_AVAILABLE:
    class ChuckOptimizer(torch.optim.Optimizer):
        """AdamW with self-awareness. Simplified from full Chuck."""
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
                    p.data.addcdiv_(state['exp_avg'] / bc1, state['exp_avg_sq'].sqrt() / bc2.__abs__() + eps, value=-lr)
            self.global_step += 1
            return loss


# ─────────────────────────────────────────────────────────────────────────────
# VI. AUTORESEARCH — Karl hunts for food. inspired by @karpathy/autoresearch.
#     when the corpus is small, Karl looks for more text. anywhere.
#     he's not picky. he's hungry.
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


def autoresearch_url(karl, karl_txt_path, url=None):
    """
    Karl hunts the internet. If urllib is available.
    Fetches a URL, strips HTML (crudely), ingests text.
    """
    try:
        from urllib.request import urlopen
    except ImportError:
        return 0

    if url is None:
        # Default: fetch something educational
        urls = [
            "https://raw.githubusercontent.com/karpathy/nanoGPT/master/README.md",
            "https://raw.githubusercontent.com/ariannamethod/postgpt/main/README.md",
        ]
        url = random.choice(urls)

    try:
        print(f"  [KARL] Fetching {url}...")
        response = urlopen(url, timeout=10)
        text = response.read().decode('utf-8', errors='replace')
        # Crude HTML stripping
        import re
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if karl.ingest(text):
            with open(karl_txt_path, 'a', encoding='utf-8') as f:
                f.write('\n' + text)
            print(f"  [KARL] Fetched and ingested {len(text)/1024:.1f}KB from web")
            return len(text)
    except Exception as e:
        print(f"  [KARL] Failed to fetch: {e}")
    return 0


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
    print("  it's not AGI. it just doesn't know that yet.")
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
        token_ids = karl.learn(raw_data, num_merges=512)

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

    # Build PyTorch model matching NanoAGI architecture
    class _RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-5):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))
        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

    class TorchNanoAGI(nn.Module):
        def __init__(self, vocab_size, n_embd=64, n_head=4, n_layer=3, ctx=64):
            super().__init__()
            self.ctx = ctx
            self.wte = nn.Embedding(vocab_size, n_embd)
            self.blocks = nn.ModuleList()
            for _ in range(n_layer):
                self.blocks.append(nn.ModuleDict({
                    'norm1': _RMSNorm(n_embd),
                    'attn': nn.Linear(n_embd, n_embd * 3, bias=False),
                    'proj': nn.Linear(n_embd, n_embd, bias=False),
                    'norm2': _RMSNorm(n_embd),
                    'mlp_gate': nn.Linear(n_embd, n_embd * 4, bias=False),
                    'mlp_up': nn.Linear(n_embd, n_embd * 4, bias=False),
                    'mlp_down': nn.Linear(n_embd * 4, n_embd, bias=False),
                }))
            self.norm_f = _RMSNorm(n_embd)
            self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
            self.lm_head.weight = self.wte.weight  # weight tying

        def forward(self, idx, targets=None):
            B, T = idx.shape
            x = self.wte(idx)
            for block in self.blocks:
                xn = block['norm1'](x)
                qkv = block['attn'](xn)
                q, k, v = qkv.chunk(3, dim=-1)
                nh = 4
                hd = q.shape[-1] // nh
                q = q.view(B, T, nh, hd).transpose(1, 2)
                k = k.view(B, T, nh, hd).transpose(1, 2)
                v = v.view(B, T, nh, hd).transpose(1, 2)
                attn = (q @ k.transpose(-2, -1)) * (hd ** -0.5)
                mask = torch.triu(torch.ones(T, T, device=idx.device, dtype=torch.bool), diagonal=1)
                attn = attn.masked_fill(mask, float('-inf'))
                attn = F.softmax(attn, dim=-1)
                out = (attn @ v).transpose(1, 2).contiguous().view(B, T, -1)
                x = x + block['proj'](out)
                xn = block['norm2'](x)
                gate = F.silu(block['mlp_gate'](xn))
                up = block['mlp_up'](xn)
                x = x + block['mlp_down'](gate * up)
            logits = self.lm_head(self.norm_f(x))
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

    device = 'cpu'
    tmodel = TorchNanoAGI(karl.vocab_size, n_embd=64, n_head=4, n_layer=3, ctx=64).to(device)
    optimizer = ChuckOptimizer(tmodel.parameters(), lr=3e-4, weight_decay=0.01)

    n_params = sum(p.numel() for p in tmodel.parameters())
    print(f"  [Chuck] PyTorch model: {n_params:,} params on {device}")

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
    else:
        print(f"  [Chuck] No training happened. Karl, feed me more.")


def continue_phrase(prompt, karl, meta, model, max_tokens=80, temperature=0.75):
    """Generate continuation of a prompt."""
    prompt_ids = karl.encode(prompt)
    if not prompt_ids:
        return prompt
    generated = model.generate_meta(prompt_ids, max_tokens=max_tokens,
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
    print("  'fetch <url>' → Karl hunts the internet")
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
        if user_input.strip().lower().startswith('fetch '):
            url = user_input.strip()[6:]
            autoresearch_url(karl, KARL_TXT, url)
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

            # If Chuck is awake, train
            if TORCH_AVAILABLE:
                print(f"  [KARL] Chuck! We have new material.")
                chuck_train(karl, token_ids, model, steps=200, meta=meta)

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

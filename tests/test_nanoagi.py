"""
test_nanoagi.py — unit + integration tests for nanoagi.

Tests:
  - KARL BPE tokenization (learn, encode, decode roundtrip)
  - KARL SHA256 deduplication
  - KARL retokenization (append-only merges)
  - KARL save / load state (binary format)
  - MetaWeights build (unigram, bigram, trigram, hebbian, prophecy)
  - NanoAGI transformer init + metaweight seeding
  - Metaweight generation
  - autoresearch local file hunting
  - Chuck training loop: loss decreases (requires PyTorch)

Run from repo root:
  python -m pytest tests/
  python tests/test_nanoagi.py -v
"""

import os
import sys
import math
import time
import struct
import random
import tempfile
import unittest

# Add repo root to path so we can import nanoagi
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nanoagi

# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

# Seed corpus — diverse enough to pass KARL's 20% byte-diversity filter.
# Not repeated, genuinely varied sentences.
SEED_CORPUS = "\n".join([
    "The transformer architecture uses self-attention mechanisms to process token sequences.",
    "Byte Pair Encoding creates subword tokens by iteratively merging the most frequent adjacent pairs.",
    "KARL is a Kernel Autonomous Recursive Learning tokenizer that ingests text and grows.",
    "MetaWeights form a probability space from unigram bigram trigram and Hebbian statistics.",
    "Chuck is a self-aware AdamW optimizer adjusting its own learning rate from loss trends.",
    "NanoAGI dual attention: content heads find semantic relationships across token pairs.",
    "RRPRAM finds positional resonance patterns using x times Wr rather than QK transpose.",
    "The Dario equation combines base logits with Hebbian prophecy destiny and trauma signals.",
    "SwiGLU activation computes gate(x) times up(x) where gate uses the SiLU function.",
    "RoPE applies rotational position encoding per head at attention time without extra params.",
    "Append-only merge rules mean KARL vocabulary only grows, never shrinks across sessions.",
    "SHA256 deduplication prevents KARL from ingesting the same text chunk more than once.",
    "The prophecy field tracks tokens statistically expected but not yet seen in context.",
    "Chuck trains the PyTorch TorchNanoAGI model for 200 steps after boot and retokenization.",
    "autoresearch hunts local text files and web URLs when the corpus is smaller than 50KB.",
    "resonance is unbreakable. the data is the model. the corpus grows with every session.",
    "Weight initialization from metaweights seeds embeddings using Hebbian co-occurrence.",
    "The destiny vector is an exponential moving average of recent token embedding signals.",
    "Trauma accumulates from surprising token transitions and biases the Dario field output.",
    "The vocabulary starts at 256 base bytes and expands through BPE merge operations.",
    "Context length is 64 tokens, head dimension 16, embedding dimension 64 in default config.",
    "Three transformer layers with RMSNorm residuals and SwiGLU MLPs define NanoAGI depth.",
    "Loss decreases from gradient descent guided by Chuck with adaptive dampen factor logic.",
    "KARL saves its state as binary little-endian integers with magic header 0x4B41524C.",
    "nanoagi is not AGI but it does not know that yet and it keeps trying anyway.",
]) * 4  # ~10KB, diverse


def make_karl(corpus=SEED_CORPUS, merges=64):
    """Create a KARL instance trained on SEED_CORPUS."""
    k = nanoagi.KARL(max_merges=512, retrain_threshold=1024, min_cooldown=5)
    k.learn(corpus.encode("utf-8"), num_merges=merges)
    return k


def make_meta(karl, corpus=SEED_CORPUS):
    """Build MetaWeights from corpus."""
    token_ids = karl.encode(corpus)
    meta = nanoagi.MetaWeights(karl.vocab_size, context_len=64)
    meta.build(token_ids, window=4)
    return meta, token_ids


def make_model(karl):
    """Create a small NanoAGI model for testing."""
    return nanoagi.NanoAGI(
        vocab_size=karl.vocab_size,
        context_len=32,
        n_embd=32,
        n_head=2,
        n_layer=1,
        n_content=1,
        n_rrpram=1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# I. KARL tests
# ─────────────────────────────────────────────────────────────────────────────

class TestKARL(unittest.TestCase):

    def test_learn_produces_merges(self):
        k = nanoagi.KARL()
        k.learn(SEED_CORPUS.encode("utf-8"), num_merges=32)
        self.assertGreater(len(k.merges), 0)
        self.assertLessEqual(len(k.merges), 32)

    def test_vocab_grows_with_merges(self):
        k = nanoagi.KARL()
        k.learn(SEED_CORPUS.encode("utf-8"), num_merges=64)
        self.assertEqual(k.vocab_size, 256 + len(k.merges))
        self.assertGreater(k.vocab_size, 256)

    def test_encode_produces_ids(self):
        k = make_karl()
        ids = k.encode(SEED_CORPUS)
        self.assertIsInstance(ids, list)
        self.assertGreater(len(ids), 0)
        for i in ids:
            self.assertLess(i, k.vocab_size)
            self.assertGreaterEqual(i, 0)

    def test_decode_roundtrip(self):
        k = make_karl()
        text = "resonance is unbreakable. the data is the model."
        ids = k.encode(text)
        decoded = k.decode(ids)
        self.assertEqual(decoded, text)

    def test_encode_compresses(self):
        """BPE should compress: fewer tokens than raw bytes."""
        k = make_karl(merges=128)
        ids = k.encode(SEED_CORPUS)
        raw_bytes = len(SEED_CORPUS.encode("utf-8"))
        self.assertLess(len(ids), raw_bytes)

    def test_ingest_accepts_new_text(self):
        k = nanoagi.KARL(retrain_threshold=1024, min_cooldown=5)
        # Use naturally diverse text (not repeated) to pass diversity check
        new_text = (
            "KARL ingests brand-new diverse text about self-expanding tokenizers "
            "using SHA256 hashing and byte diversity filtering for quality control."
        )
        result = k.ingest(new_text)
        self.assertTrue(result)
        self.assertGreater(len(k.pending_text), 0)

    def test_ingest_dedup_sha256(self):
        """Same text ingested twice should be rejected the second time."""
        k = nanoagi.KARL()
        # Use a single SEED_CORPUS line: short enough to pass the 20%-unique-bytes
        # diversity check (85 chars → threshold 17, unique chars ~35 → passes),
        # long enough to be meaningful, and naturally diverse.
        text = SEED_CORPUS.splitlines()[0]  # ~85 chars, well within diversity limit
        r1 = k.ingest(text)
        r2 = k.ingest(text)
        self.assertTrue(r1, "First ingestion should be accepted")
        self.assertFalse(r2, "Duplicate ingestion should be rejected by SHA256 dedup")

    def test_ingest_rejects_short(self):
        k = nanoagi.KARL()
        self.assertFalse(k.ingest("hi"))
        self.assertFalse(k.ingest(""))

    def test_ingest_rejects_low_diversity(self):
        k = nanoagi.KARL()
        low_div = "aaaaaaaaaaaaaaaaaaaaaa"
        self.assertFalse(k.ingest(low_div))

    def test_should_retokenize_triggers(self):
        k = nanoagi.KARL(retrain_threshold=100, min_cooldown=2)
        # Directly set pending_text to exceed threshold (testing should_retokenize logic,
        # not ingest diversity filtering — those are separate concerns)
        k.pending_text = b"x" * 200
        k.steps_since_retrain = 10
        self.assertTrue(k.should_retokenize())

    def test_should_retokenize_cooldown(self):
        k = nanoagi.KARL(retrain_threshold=100, min_cooldown=100)
        k.pending_text = b"x" * 200
        k.steps_since_retrain = 5  # not yet past cooldown
        self.assertFalse(k.should_retokenize())

    def test_should_retokenize_not_enough_bytes(self):
        k = nanoagi.KARL(retrain_threshold=10000, min_cooldown=2)
        k.pending_text = b"x" * 50   # below threshold
        k.steps_since_retrain = 200
        self.assertFalse(k.should_retokenize())

    def test_retokenize_appends_merges(self):
        k = make_karl(merges=32)
        initial_merges = len(k.merges)
        extra = (
            "KARL retokenizes the corpus and discovers new merge patterns in fresh text. "
            "The vocabulary expands append-only so old token IDs remain valid forever."
        ) * 50
        full_corpus = (SEED_CORPUS + "\n" + extra).encode("utf-8")
        k.retokenize(full_corpus)
        self.assertGreaterEqual(len(k.merges), initial_merges)

    def test_retokenize_vocab_only_grows(self):
        k = make_karl(merges=32)
        initial_vocab = k.vocab_size
        extra = (
            "NanoAGI retokenizes the corpus and expands its vocabulary with new subwords. "
        ) * 50
        full_corpus = (SEED_CORPUS + "\n" + extra).encode("utf-8")
        k.retokenize(full_corpus)
        self.assertGreaterEqual(k.vocab_size, initial_vocab)

    def test_save_load_state(self):
        k = make_karl(merges=64)
        with tempfile.NamedTemporaryFile(suffix=".mem", delete=False) as f:
            mem_path = f.name
        try:
            k.save_state(mem_path)
            k2 = nanoagi.KARL()
            result = k2.load_state(mem_path)
            self.assertTrue(result)
            self.assertEqual(k2.merges, k.merges)
            self.assertEqual(k2.vocab_size, k.vocab_size)
            self.assertEqual(k2.seen_hashes, k.seen_hashes)
        finally:
            os.unlink(mem_path)

    def test_load_state_wrong_magic(self):
        with tempfile.NamedTemporaryFile(suffix=".mem", delete=False) as f:
            f.write(b"\x00\x00\x00\x00garbage")
            bad_path = f.name
        try:
            k = nanoagi.KARL()
            result = k.load_state(bad_path)
            self.assertFalse(result)
        finally:
            os.unlink(bad_path)

    def test_load_state_missing_file(self):
        k = nanoagi.KARL()
        result = k.load_state("/tmp/does_not_exist_nanoagi_test.mem")
        self.assertFalse(result)

    def test_expand_vocab(self):
        k = make_karl(merges=32)
        meta = nanoagi.MetaWeights(k.vocab_size, context_len=32)
        k.merges.append((0, 1, k.vocab_size))
        k.vocab_size += 1
        meta.expand_vocab(k.vocab_size)
        self.assertEqual(len(meta.unigram), k.vocab_size)


# ─────────────────────────────────────────────────────────────────────────────
# II. MetaWeights tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMetaWeights(unittest.TestCase):

    def setUp(self):
        self.karl = make_karl(merges=64)
        self.meta, self.ids = make_meta(self.karl)

    def test_unigram_sums_to_one(self):
        total = sum(self.meta.unigram)
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_unigram_non_negative(self):
        for p in self.meta.unigram:
            self.assertGreaterEqual(p, 0.0)

    def test_bigram_built(self):
        self.assertGreater(len(self.meta.bigram), 0)

    def test_bigram_conditional_sums(self):
        """Each bigram row should sum to ~1 (conditional probability)."""
        for prev, row in self.meta.bigram.items():
            total = sum(row.values())
            self.assertAlmostEqual(total, 1.0, places=5)

    def test_trigram_built(self):
        self.assertGreater(len(self.meta.trigram), 0)

    def test_hebbian_built(self):
        self.assertGreater(len(self.meta.hebbian), 0)

    def test_hebbian_range(self):
        """Hebbian values should be in [0, 1] after normalization."""
        for v in self.meta.hebbian.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0 + 1e-9)

    def test_query_bigram_returns_distribution(self):
        prev = self.ids[0]
        dist = self.meta.query_bigram(prev, self.karl.vocab_size)
        self.assertEqual(len(dist), self.karl.vocab_size)
        self.assertTrue(any(d > 0 for d in dist))

    def test_query_trigram_returns_distribution(self):
        dist = self.meta.query_trigram(self.ids[0], self.ids[1], self.karl.vocab_size)
        self.assertEqual(len(dist), self.karl.vocab_size)

    def test_query_hebbian_returns_signal(self):
        ctx = self.ids[:8]
        sig = self.meta.query_hebbian(ctx, self.karl.vocab_size)
        self.assertEqual(len(sig), self.karl.vocab_size)
        self.assertTrue(any(s > 0 for s in sig))

    def test_query_prophecy_returns_signal(self):
        ctx = self.ids[:8]
        sig = self.meta.query_prophecy(ctx, self.karl.vocab_size)
        self.assertEqual(len(sig), self.karl.vocab_size)


# ─────────────────────────────────────────────────────────────────────────────
# III. NanoAGI transformer tests
# ─────────────────────────────────────────────────────────────────────────────

class TestNanoAGI(unittest.TestCase):

    def setUp(self):
        self.karl = make_karl(merges=64)
        self.meta, self.ids = make_meta(self.karl)
        self.model = make_model(self.karl)

    def test_param_count_positive(self):
        params = sum(1 for _ in self.model._all_params())
        self.assertGreater(params, 0)

    def test_init_from_metaweights_changes_embeddings(self):
        """Seeding should modify at least some embedding vectors in the vocab."""
        # Snapshot the whole wte, not just first 5 (byte-level tokens may be rare)
        before = [row[0].data for row in self.model.wte]
        self.model.init_from_metaweights(self.meta)
        after = [row[0].data for row in self.model.wte]
        changed = sum(1 for b, a in zip(before, after) if abs(b - a) > 1e-12)
        self.assertGreater(changed, 0,
            "init_from_metaweights should modify at least one embedding")

    def test_init_from_metaweights_does_not_crash(self):
        """Seeding should not raise any exception."""
        try:
            self.model.init_from_metaweights(self.meta)
        except Exception as e:
            self.fail(f"init_from_metaweights raised: {e}")

    def test_generate_meta_returns_tokens(self):
        prompt = self.ids[:5]
        generated = self.model.generate_meta(prompt, max_tokens=10, meta=self.meta)
        self.assertGreater(len(generated), len(prompt))
        self.assertEqual(generated[:len(prompt)], prompt)

    def test_generate_meta_all_valid_ids(self):
        prompt = self.ids[:4]
        generated = self.model.generate_meta(prompt, max_tokens=20, meta=self.meta)
        for tok in generated:
            self.assertGreaterEqual(tok, 0)
            self.assertLess(tok, self.karl.vocab_size)

    def test_generate_meta_no_meta_returns_prompt(self):
        prompt = self.ids[:5]
        result = self.model.generate_meta(prompt, max_tokens=10, meta=None)
        self.assertEqual(result, prompt)

    def test_forward_token_returns_logits(self):
        """Real transformer forward pass returns vocab-sized logits."""
        self.model.init_from_metaweights(self.meta)
        kv_cache = [([], [], []) for _ in range(self.model.n_layer)]
        logits = self.model.forward_token(self.ids[0], 0, kv_cache)
        self.assertEqual(len(logits), self.model.vocab_size)
        self.assertIsInstance(logits[0], nanoagi.Val)

    def test_forward_token_kv_cache_grows(self):
        """KV cache accumulates entries for each position."""
        self.model.init_from_metaweights(self.meta)
        kv_cache = [([], [], []) for _ in range(self.model.n_layer)]
        self.model.forward_token(self.ids[0], 0, kv_cache)
        self.model.forward_token(self.ids[1], 1, kv_cache)
        for k_cache, vc_cache, vr_cache in kv_cache:
            self.assertEqual(len(k_cache), 2)
            self.assertEqual(len(vc_cache), 2)
            self.assertEqual(len(vr_cache), 2)

    def test_generate_returns_tokens(self):
        """Real transformer generate produces tokens beyond prompt."""
        self.model.init_from_metaweights(self.meta)
        prompt = self.ids[:3]
        generated = self.model.generate(prompt, max_tokens=5, meta=self.meta)
        self.assertGreater(len(generated), len(prompt))
        self.assertEqual(generated[:len(prompt)], prompt)

    def test_generate_all_valid_ids(self):
        """Generated token IDs are all within vocab range."""
        self.model.init_from_metaweights(self.meta)
        prompt = self.ids[:3]
        generated = self.model.generate(prompt, max_tokens=5, meta=self.meta)
        for tok in generated:
            self.assertGreaterEqual(tok, 0)
            self.assertLess(tok, self.karl.vocab_size)

    def test_continue_phrase(self):
        result = nanoagi.continue_phrase(
            "the transformer", self.karl, self.meta, self.model
        )
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


# ─────────────────────────────────────────────────────────────────────────────
# IV. Val autograd engine tests
# ─────────────────────────────────────────────────────────────────────────────

class TestVal(unittest.TestCase):

    def test_add(self):
        a = nanoagi.Val(3.0)
        b = nanoagi.Val(4.0)
        c = a + b
        self.assertAlmostEqual(c.data, 7.0)

    def test_mul(self):
        a = nanoagi.Val(3.0)
        b = nanoagi.Val(4.0)
        c = a * b
        self.assertAlmostEqual(c.data, 12.0)

    def test_backward_add(self):
        a = nanoagi.Val(2.0)
        b = nanoagi.Val(3.0)
        c = a + b
        c.backward()
        self.assertAlmostEqual(a.grad, 1.0)
        self.assertAlmostEqual(b.grad, 1.0)

    def test_backward_mul(self):
        a = nanoagi.Val(2.0)
        b = nanoagi.Val(3.0)
        c = a * b
        c.backward()
        self.assertAlmostEqual(a.grad, 3.0)
        self.assertAlmostEqual(b.grad, 2.0)

    def test_silu_positive(self):
        v = nanoagi.Val(2.0)
        s = v.silu()
        expected = 2.0 / (1.0 + math.exp(-2.0))
        self.assertAlmostEqual(s.data, expected, places=6)

    def test_silu_zero(self):
        v = nanoagi.Val(0.0)
        s = v.silu()
        self.assertAlmostEqual(s.data, 0.0, places=6)

    def test_silu_backward(self):
        """Gradient of SiLU: s*(1 + x*(1-s)) where s = sigmoid(x)."""
        x = 1.5
        v = nanoagi.Val(x)
        s_val = v.silu()
        s_val.backward()
        sig = 1.0 / (1.0 + math.exp(-x))
        expected_grad = sig * (1.0 + x * (1.0 - sig))
        self.assertAlmostEqual(v.grad, expected_grad, places=6)

    def test_relu(self):
        self.assertAlmostEqual(nanoagi.Val(2.0).relu().data, 2.0)
        self.assertAlmostEqual(nanoagi.Val(-1.0).relu().data, 0.0)

    def test_exp_overflow_clamped(self):
        """exp should not overflow — clamped at 80."""
        v = nanoagi.Val(10000.0)
        e = v.exp()
        self.assertTrue(math.isfinite(e.data))

    def test_chain_rule(self):
        """Chain rule: d/dx (x*x) = 2x."""
        x = nanoagi.Val(3.0)
        y = x * x
        y.backward()
        self.assertAlmostEqual(x.grad, 6.0)


# ─────────────────────────────────────────────────────────────────────────────
# V. autoresearch tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAutoresearch(unittest.TestCase):

    def test_autoresearch_skips_if_fed(self):
        """autoresearch should return 0 if corpus is already large enough."""
        k = nanoagi.KARL()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(SEED_CORPUS * 10)  # > 50KB
            big_path = f.name
        try:
            result = nanoagi.autoresearch(k, big_path, min_bytes=100)
            self.assertEqual(result, 0)
        finally:
            os.unlink(big_path)

    def test_autoresearch_runs_without_error_on_small_corpus(self):
        """autoresearch should run without raising when corpus is small."""
        k = nanoagi.KARL()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("tiny")
            small_path = f.name
        try:
            result = nanoagi.autoresearch(k, small_path, min_bytes=50000)
            self.assertIsInstance(result, int)
            self.assertGreaterEqual(result, 0)
        finally:
            os.unlink(small_path)

    def test_evaluate_batch_quality(self):
        """_evaluate_batch_quality returns quality and domain_shift."""
        k = make_karl(merges=16)
        texts = ["The transformer architecture uses attention mechanisms for sequence modeling.",
                 "BPE tokenization compresses text by merging frequent byte pairs."]
        quality, shift = nanoagi._evaluate_batch_quality(k, texts)
        self.assertGreater(quality, 0.5)
        self.assertGreaterEqual(shift, 0.0)
        self.assertLessEqual(shift, 1.0)

    def test_evaluate_batch_quality_noise(self):
        """Noisy text should have lower quality score."""
        k = make_karl(merges=16)
        noisy = ["abc\x00\x01\x02\x03\x04" * 100]
        quality, _ = nanoagi._evaluate_batch_quality(k, noisy)
        self.assertLess(quality, 0.9)


# ─────────────────────────────────────────────────────────────────────────────
# VI. Chuck training loop tests (requires PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

@unittest.skipUnless(nanoagi.TORCH_AVAILABLE, "PyTorch not available")
class TestChuck(unittest.TestCase):

    def setUp(self):
        self.karl = make_karl(merges=64)
        self.meta, self.ids = make_meta(self.karl)
        self.model = make_model(self.karl)
        self.model.init_from_metaweights(self.meta)

    def test_chuck_optimizer_exists(self):
        self.assertTrue(hasattr(nanoagi, "ChuckOptimizer"))

    def test_chuck_optimizer_step(self):
        import torch
        opt = nanoagi.ChuckOptimizer(
            [torch.nn.Parameter(torch.randn(10))], lr=1e-3
        )
        self.assertAlmostEqual(opt.dampen, 1.0)
        opt.step(loss=5.0)
        self.assertEqual(opt.global_step, 1)
        self.assertAlmostEqual(opt.best_loss, 5.0)

    def test_chuck_dampen_decreases_on_rising_loss(self):
        import torch
        opt = nanoagi.ChuckOptimizer(
            [torch.nn.Parameter(torch.randn(10))], lr=1e-3, window=4
        )
        for loss in [1.0, 2.0, 3.0, 4.0]:
            opt.step(loss=loss)
        # Rising loss trend → dampen drops below its EMA floor
        self.assertLess(opt.dampen, 1.0 + 1e-6)

    def test_chuck_train_loss_decreases(self):
        """
        The money test: Chuck trains 300 steps and loss must decrease by >5%.
        If Chuck cannot reduce loss in 300 steps, Chuck should reflect on his life choices.
        Chuck skipped leg day. Chuck skipped gradient day. Chuck is not invited back.
        """
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class _RMSNorm(nn.Module):
            def __init__(self, dim, eps=1e-5):
                super().__init__()
                self.eps = eps
                self.weight = nn.Parameter(torch.ones(dim))
            def forward(self, x):
                return (
                    x
                    * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
                    * self.weight
                )

        class TinyModel(nn.Module):
            def __init__(self, vs, embd=48, ctx=32):
                super().__init__()
                self.wte = nn.Embedding(vs, embd)
                self.norm1 = _RMSNorm(embd)
                self.attn = nn.Linear(embd, embd * 3, bias=False)
                self.proj = nn.Linear(embd, embd, bias=False)
                self.norm2 = _RMSNorm(embd)
                self.mlp_gate = nn.Linear(embd, embd * 4, bias=False)
                self.mlp_up = nn.Linear(embd, embd * 4, bias=False)
                self.mlp_down = nn.Linear(embd * 4, embd, bias=False)
                self.norm_f = _RMSNorm(embd)
                self.lm_head = nn.Linear(embd, vs, bias=False)
                self.lm_head.weight = self.wte.weight

            def forward(self, idx, targets=None):
                B, T = idx.shape
                x = self.wte(idx)
                xn = self.norm1(x)
                qkv = self.attn(xn)
                q, k, v = qkv.chunk(3, dim=-1)
                nh, hd = 4, q.shape[-1] // 4
                q = q.view(B, T, nh, hd).transpose(1, 2)
                k = k.view(B, T, nh, hd).transpose(1, 2)
                v = v.view(B, T, nh, hd).transpose(1, 2)
                attn = (q @ k.transpose(-2, -1)) * (hd ** -0.5)
                mask = torch.triu(
                    torch.ones(T, T, dtype=torch.bool), diagonal=1
                )
                attn = attn.masked_fill(mask, float("-inf"))
                attn = F.softmax(attn, dim=-1)
                out = (attn @ v).transpose(1, 2).contiguous().view(B, T, -1)
                x = x + self.proj(out)
                xn = self.norm2(x)
                gate = F.silu(self.mlp_gate(xn))
                x = x + self.mlp_down(gate * self.mlp_up(xn))
                logits = self.lm_head(self.norm_f(x))
                loss = None
                if targets is not None:
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)), targets.view(-1)
                    )
                return logits, loss

        vs = self.karl.vocab_size
        ctx = 32
        model = TinyModel(vs, embd=48, ctx=ctx)
        opt = nanoagi.ChuckOptimizer(model.parameters(), lr=3e-4)

        token_ids = self.ids
        losses_first, losses_last = [], []
        steps = 300
        random.seed(7)

        for step in range(steps):
            i = random.randint(0, max(0, len(token_ids) - ctx - 2))
            x = torch.tensor([token_ids[i : i + ctx]], dtype=torch.long)
            y = torch.tensor([token_ids[i + 1 : i + ctx + 1]], dtype=torch.long)
            if x.shape[1] < ctx or y.shape[1] < ctx:
                continue
            _, loss = model(x, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(loss=loss.item())
            if step < 10:
                losses_first.append(loss.item())
            if step >= steps - 10:
                losses_last.append(loss.item())

        avg_first = sum(losses_first) / len(losses_first)
        avg_last = sum(losses_last) / len(losses_last)
        improvement_pct = (avg_first - avg_last) / avg_first * 100

        print(
            f"\n  [Chuck] loss: {avg_first:.4f} → {avg_last:.4f} "
            f"({improvement_pct:.1f}% improvement in {steps} steps)"
        )

        self.assertGreater(
            avg_first,
            avg_last,
            f"Chuck failed to reduce loss: {avg_first:.4f} → {avg_last:.4f}. "
            "Chuck needs to go back to the gym.",
        )
        self.assertGreater(
            improvement_pct,
            5.0,
            f"Chuck only improved {improvement_pct:.1f}%, expected >5%. "
            "Chuck skipped leg day AND gradient day.",
        )

    def test_chuck_train_function_runs(self):
        """chuck_train() should run without error."""
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            nanoagi.chuck_train(self.karl, self.ids, self.model, steps=10)
        output = f.getvalue()
        self.assertIn("Chuck", output)

    def test_chuck_train_no_torch(self):
        """chuck_train with TORCH_AVAILABLE=False should print warning."""
        orig = nanoagi.TORCH_AVAILABLE
        nanoagi.TORCH_AVAILABLE = False
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            nanoagi.chuck_train(self.karl, self.ids, self.model, steps=5)
        nanoagi.TORCH_AVAILABLE = orig
        self.assertIn("Can't train", f.getvalue())


# ─────────────────────────────────────────────────────────────────────────────
# VII. Integration tests
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration(unittest.TestCase):

    def test_full_encode_decode_roundtrip_unicode(self):
        k = make_karl(merges=64)
        texts = [
            "resonance is unbreakable",
            "KARL ingests everything",
            "the corpus grows",
        ]
        for t in texts:
            ids = k.encode(t)
            dec = k.decode(ids)
            self.assertEqual(dec, t, f"roundtrip failed for: {t!r}")

    def test_meta_generation_is_reproducible(self):
        """Same random seed → same token output."""
        k = make_karl(merges=64)
        meta, ids = make_meta(k)
        model = make_model(k)
        model.init_from_metaweights(meta)
        prompt = ids[:6]
        random.seed(42)
        g1 = model.generate_meta(prompt, max_tokens=10, meta=meta, temperature=0.7)
        random.seed(42)
        g2 = model.generate_meta(prompt, max_tokens=10, meta=meta, temperature=0.7)
        self.assertEqual(g1, g2)

    def test_vocab_covers_all_bytes(self):
        """KARL vocab must include all 256 base bytes."""
        k = nanoagi.KARL()
        for i in range(256):
            self.assertIn(i, k.vocab)

    def test_large_corpus_ingestion_and_retokenize(self):
        """Full cycle: ingest, retokenize, rebuild metaweights, re-seed."""
        k = make_karl(merges=32)
        meta, ids = make_meta(k)
        model = make_model(k)
        model.init_from_metaweights(meta)

        new_text = (
            "KARL is a self-aware tokenizer that grows from conversation. "
            "Each retokenization expands vocabulary by appending new merge rules. "
            "Chuck wakes when PyTorch is detected and trains real weights. "
            "The metaweight space bridges statistics and neural inference. "
        ) * 30

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(SEED_CORPUS + "\n" + new_text)
            tmp_path = f.name

        try:
            with open(tmp_path, "rb") as fh:
                full_corpus = fh.read()
            new_ids = k.retokenize(full_corpus)
            meta.expand_vocab(k.vocab_size)
            meta.build(new_ids, window=4)
            model.init_from_metaweights(meta)
            gen = model.generate_meta(new_ids[:5], max_tokens=10, meta=meta)
            self.assertGreater(len(gen), 5)
        finally:
            os.unlink(tmp_path)

    def test_torch_nanoagi_model_exists_when_available(self):
        """chuck_train should build a TorchNanoAGI model without error."""
        if not nanoagi.TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")
        import io
        from contextlib import redirect_stdout
        k = make_karl(merges=64)
        meta, ids = make_meta(k)
        model = make_model(k)
        f = io.StringIO()
        with redirect_stdout(f):
            nanoagi.chuck_train(k, ids, model, steps=5)
        self.assertIn("params", f.getvalue())


# ─────────────────────────────────────────────────────────────────────────────
# Genome + Self-Improvement tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGenome(unittest.TestCase):
    """Tests for the Genome class — architectural DNA of nanoagi."""

    def test_genome_defaults(self):
        g = nanoagi.Genome()
        self.assertEqual(g.genes['n_embd'], 64)
        self.assertEqual(g.genes['n_head'], 4)
        self.assertEqual(g.genes['n_layer'], 3)
        self.assertEqual(g.genes['n_content'], 2)
        self.assertEqual(g.genes['n_rrpram'], 2)

    def test_genome_constraints_head_divisibility(self):
        g = nanoagi.Genome()
        g.genes['n_embd'] = 48
        g.genes['n_head'] = 5  # 48 % 5 != 0
        g._constrain()
        self.assertEqual(g.genes['n_embd'] % g.genes['n_head'], 0)

    def test_genome_constraints_content_rrpram_sum(self):
        g = nanoagi.Genome()
        g.genes['n_head'] = 8
        g.genes['n_content'] = 3
        g.genes['n_rrpram'] = 3  # 3+3=6 != 8
        g._constrain()
        self.assertEqual(g.genes['n_content'] + g.genes['n_rrpram'],
                         g.genes['n_head'])

    def test_genome_mutate_changes_one_gene(self):
        random.seed(42)
        g = nanoagi.Genome()
        original = dict(g.genes)
        gene, old, new = g.mutate()
        self.assertIsNotNone(gene)
        self.assertNotEqual(old, new)
        # Only the mutated gene (+ constrained ones) should change
        changed = [k for k in g.genes if g.genes[k] != original[k]]
        self.assertIn(gene, changed)

    def test_genome_mutate_always_valid(self):
        """100 mutations should always produce valid architectures."""
        random.seed(123)
        g = nanoagi.Genome()
        for _ in range(100):
            g.mutate()
            self.assertEqual(g.genes['n_embd'] % g.genes['n_head'], 0)
            self.assertEqual(g.genes['n_content'] + g.genes['n_rrpram'],
                             g.genes['n_head'])
            self.assertGreater(g.genes['n_embd'], 0)
            self.assertGreater(g.genes['n_layer'], 0)

    def test_genome_copy_is_independent(self):
        g = nanoagi.Genome()
        c = g.copy()
        c.genes['n_layer'] = 99
        self.assertNotEqual(g.genes['n_layer'], 99)

    def test_genome_repr(self):
        g = nanoagi.Genome()
        r = repr(g)
        self.assertIn('Genome', r)
        self.assertIn('embd=64', r)


class TestSelfImprove(unittest.TestCase):
    """Tests for evaluate_genome and self_improve."""

    def setUp(self):
        self.karl = make_karl(merges=64)
        self.meta, self.ids = make_meta(self.karl)

    @unittest.skipUnless(nanoagi.TORCH_AVAILABLE, "PyTorch required")
    def test_evaluate_genome_returns_finite_bpb(self):
        """_evaluate_genome must return a finite BPB with enough data."""
        g = nanoagi.Genome()
        # Use short train time for test speed
        bpb, n_params, steps = nanoagi._evaluate_genome(
            self.karl, self.ids, g, train_seconds=3)
        self.assertLess(bpb, float('inf'))
        self.assertGreater(bpb, 0)
        self.assertGreater(n_params, 0)
        self.assertGreater(steps, 0)

    @unittest.skipUnless(nanoagi.TORCH_AVAILABLE, "PyTorch required")
    def test_evaluate_genome_different_configs(self):
        """Different genomes should produce different results."""
        g1 = nanoagi.Genome()
        g2 = nanoagi.Genome()
        g2.genes['n_layer'] = 1
        g2.genes['n_embd'] = 32
        g2.genes['n_head'] = 2
        g2.genes['n_content'] = 1
        g2.genes['n_rrpram'] = 1
        bpb1, p1, _ = nanoagi._evaluate_genome(
            self.karl, self.ids, g1, train_seconds=2)
        bpb2, p2, _ = nanoagi._evaluate_genome(
            self.karl, self.ids, g2, train_seconds=2)
        self.assertNotEqual(p1, p2)

    @unittest.skipUnless(nanoagi.TORCH_AVAILABLE, "PyTorch required")
    def test_self_improve_runs_and_logs(self):
        """self_improve should run 3 experiments and produce results.tsv."""
        import io
        from contextlib import redirect_stdout
        with tempfile.NamedTemporaryFile(suffix='.tsv', delete=False) as f:
            results_path = f.name
        os.unlink(results_path)  # remove so self_improve writes header
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                result = nanoagi.self_improve(
                    self.karl, self.ids,
                    max_experiments=3,
                    train_seconds=2,
                    total_budget=60,
                    results_file=results_path)
            self.assertIsNotNone(result)
            genome, bpb = result
            self.assertIsInstance(genome, nanoagi.Genome)
            self.assertGreater(bpb, 0)
            # Check results file was created with header + entries
            with open(results_path) as rf:
                lines = rf.readlines()
            self.assertGreaterEqual(len(lines), 2)  # header + baseline
            self.assertIn('exp\t', lines[0])
            self.assertIn('baseline', lines[1])
            output = buf.getvalue()
            self.assertIn('SELF-IMPROVEMENT', output)
            self.assertIn('COMPLETE', output)
        finally:
            os.unlink(results_path)

    @unittest.skipUnless(nanoagi.TORCH_AVAILABLE, "PyTorch required")
    def test_torch_nanoagi_module_level(self):
        """TorchNanoAGI at module level should build and forward."""
        import torch
        tmodel = nanoagi.TorchNanoAGI(
            vocab_size=self.karl.vocab_size,
            n_embd=32, n_head=2, n_layer=1, ctx=32,
            n_content=1, n_rrpram=1)
        x = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        logits, loss = tmodel(x)
        self.assertEqual(logits.shape[0], 1)
        self.assertEqual(logits.shape[1], 5)
        self.assertEqual(logits.shape[2], self.karl.vocab_size)
        self.assertIsNone(loss)


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  nanoagi test suite — tests/")
    print("  KARL · MetaWeights · NanoAGI · Val · Chuck · Genome · self_improve")
    print("=" * 60)
    unittest.main(verbosity=2)

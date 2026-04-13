"""
notorch_nn.py — Python neural network API backed by libnotorch (ctypes)

Drop-in shim: replaces torch.nn, torch.optim, torch for nanoagi.
Supports RRPRAM dual attention (content + positional pattern recognition).

No PyTorch. No numpy. Just ctypes to the C library.

Usage:
    from ariannamethod.notorch_nn import *
"""

import ctypes
import os
import math
import random
import struct
import subprocess

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD libnotorch
# ═══════════════════════════════════════════════════════════════════════════════

_dir = os.path.dirname(os.path.abspath(__file__))

for ext in ['.dylib', '.so', '.dll']:
    _libpath = os.path.join(_dir, f'libnotorch{ext}')
    if os.path.exists(_libpath):
        break
else:
    _src = os.path.join(_dir, 'notorch.c')
    _libpath = os.path.join(_dir, 'libnotorch.dylib')
    if os.path.exists(_src):
        subprocess.run(['cc', '-O2', '-std=c11', '-shared', '-fPIC',
                       '-o', _libpath, _src, '-lm'], check=True)

_lib = ctypes.CDLL(_libpath)

# ═══════════════════════════════════════════════════════════════════════════════
# C FUNCTION SIGNATURES
# ═══════════════════════════════════════════════════════════════════════════════

# Tensor
_lib.nt_tensor_new.restype = ctypes.c_void_p
_lib.nt_tensor_new.argtypes = [ctypes.c_int]
_lib.nt_tensor_new2d.restype = ctypes.c_void_p
_lib.nt_tensor_new2d.argtypes = [ctypes.c_int, ctypes.c_int]
_lib.nt_tensor_free.argtypes = [ctypes.c_void_p]
_lib.nt_tensor_xavier.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_lib.nt_tensor_fill.argtypes = [ctypes.c_void_p, ctypes.c_float]
_lib.nt_tensor_rand.argtypes = [ctypes.c_void_p, ctypes.c_float]

# Tape
_lib.nt_tape_start.restype = None
_lib.nt_tape_clear.restype = None
_lib.nt_tape_param.restype = ctypes.c_int
_lib.nt_tape_param.argtypes = [ctypes.c_void_p]
_lib.nt_tape_no_decay.argtypes = [ctypes.c_int]
_lib.nt_tape_backward.argtypes = [ctypes.c_int]
_lib.nt_tape_clip_grads.restype = ctypes.c_float
_lib.nt_tape_clip_grads.argtypes = [ctypes.c_float]
_lib.nt_tape_chuck_step.argtypes = [ctypes.c_float, ctypes.c_float]
_lib.nt_tape_adam_step.argtypes = [ctypes.c_float]
_lib.nt_tape_adamw_step.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
_lib.nt_train_mode.argtypes = [ctypes.c_int]

# Ops
_lib.nt_seq_embedding.restype = ctypes.c_int
_lib.nt_seq_embedding.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.nt_seq_rmsnorm.restype = ctypes.c_int
_lib.nt_seq_rmsnorm.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.nt_seq_linear.restype = ctypes.c_int
_lib.nt_seq_linear.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.nt_mh_causal_attention.restype = ctypes.c_int
_lib.nt_mh_causal_attention.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.nt_rrpram_attention.restype = ctypes.c_int
_lib.nt_rrpram_attention.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                      ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.nt_concat.restype = ctypes.c_int
_lib.nt_concat.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.nt_rope.restype = ctypes.c_int
_lib.nt_rope.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.nt_silu.restype = ctypes.c_int
_lib.nt_silu.argtypes = [ctypes.c_int]
_lib.nt_add.restype = ctypes.c_int
_lib.nt_add.argtypes = [ctypes.c_int, ctypes.c_int]
_lib.nt_mul.restype = ctypes.c_int
_lib.nt_mul.argtypes = [ctypes.c_int, ctypes.c_int]
_lib.nt_seq_cross_entropy.restype = ctypes.c_int
_lib.nt_seq_cross_entropy.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]

# Record
_lib.nt_tape_record.restype = ctypes.c_int
_lib.nt_tape_record.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float]

# Save/Load
_lib.nt_save.restype = ctypes.c_int
_lib.nt_save.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
_lib.nt_load.restype = ctypes.POINTER(ctypes.c_void_p)
_lib.nt_load.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]

# RNG + tape
_lib.nt_seed.argtypes = [ctypes.c_uint64]
_lib.nt_tape_get.restype = ctypes.c_void_p

# ═══════════════════════════════════════════════════════════════════════════════
# STRUCTS
# ═══════════════════════════════════════════════════════════════════════════════

class _NtTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_float)),
        ("ndim", ctypes.c_int),
        ("shape", ctypes.c_int * 8),
        ("stride", ctypes.c_int * 8),
        ("len", ctypes.c_int),
        ("refcount", ctypes.c_int),
    ]

class _NtTapeEntry(ctypes.Structure):
    _fields_ = [
        ("output", ctypes.c_void_p),
        ("grad", ctypes.c_void_p),
        ("op", ctypes.c_int),
        ("parent1", ctypes.c_int),
        ("parent2", ctypes.c_int),
        ("parent3", ctypes.c_int),
        ("aux", ctypes.c_float),
        ("aux2", ctypes.c_float),
        ("aux3", ctypes.c_float),
        ("aux4", ctypes.c_float),
        ("is_param", ctypes.c_int),
        ("no_decay", ctypes.c_int),
    ]


def _get_ts(ptr):
    return ctypes.cast(ptr, ctypes.POINTER(_NtTensor)).contents


# ═══════════════════════════════════════════════════════════════════════════════
# TENSOR
# ═══════════════════════════════════════════════════════════════════════════════

class Tensor:
    def __init__(self, ptr, owns=True):
        self._ptr = ptr
        self._owns = owns

    @staticmethod
    def zeros(size):
        if isinstance(size, (int,)):
            return Tensor(_lib.nt_tensor_new(size))
        elif len(size) == 1:
            return Tensor(_lib.nt_tensor_new(size[0]))
        else:
            return Tensor(_lib.nt_tensor_new2d(size[0], size[1]))

    @staticmethod
    def ones(size):
        t = Tensor.zeros(size) if isinstance(size, int) else Tensor.zeros(size)
        _lib.nt_tensor_fill(t._ptr, 1.0)
        return t

    @property
    def numel(self):
        return _get_ts(self._ptr).len

    @property
    def shape(self):
        s = _get_ts(self._ptr)
        return tuple(s.shape[i] for i in range(s.ndim))

    def fill_(self, val):
        _lib.nt_tensor_fill(self._ptr, ctypes.c_float(val))
        return self

    def xavier_(self, fan_in, fan_out):
        _lib.nt_tensor_xavier(self._ptr, fan_in, fan_out)
        return self

    def rand_(self, scale):
        _lib.nt_tensor_rand(self._ptr, ctypes.c_float(scale))
        return self

    def set_data(self, flat_list):
        s = _get_ts(self._ptr)
        for i in range(min(len(flat_list), s.len)):
            s.data[i] = flat_list[i]

    def get_data(self):
        s = _get_ts(self._ptr)
        return [s.data[i] for i in range(s.len)]

    def __del__(self):
        if self._owns and self._ptr:
            _lib.nt_tensor_free(self._ptr)


class Parameter(Tensor):
    @staticmethod
    def zeros(size):
        if isinstance(size, int):
            ptr = _lib.nt_tensor_new(size)
        elif len(size) == 1:
            ptr = _lib.nt_tensor_new(size[0])
        else:
            ptr = _lib.nt_tensor_new2d(size[0], size[1])
        return Parameter(ptr)

    @staticmethod
    def ones(size):
        p = Parameter.zeros(size)
        _lib.nt_tensor_fill(p._ptr, 1.0)
        return p


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self._training = True

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self.__dict__.setdefault('_parameters', {})[name] = value
        elif isinstance(value, Module):
            self.__dict__.setdefault('_modules', {})[name] = value
        super().__setattr__(name, value)

    def parameters(self):
        for p in self._parameters.values():
            yield p
        for m in self._modules.values():
            yield from m.parameters()

    def named_parameters(self, prefix=''):
        for name, p in self._parameters.items():
            yield (f'{prefix}{name}' if prefix else name), p
        for mname, m in self._modules.items():
            yield from m.named_parameters(prefix=f'{prefix}{mname}.')

    def n_params(self):
        return sum(p.numel for p in self.parameters())

    def train(self, mode=True):
        self._training = mode
        _lib.nt_train_mode(1 if mode else 0)
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self):
        return self.train(False)


class Linear(Module):
    def __init__(self, in_f, out_f, bias=False):
        super().__init__()
        self.in_features = in_f
        self.out_features = out_f
        self.weight = Parameter.zeros((out_f, in_f))
        self.weight.xavier_(in_f, out_f)


class Embedding(Module):
    def __init__(self, num, dim):
        super().__init__()
        self.num_embeddings = num
        self.embedding_dim = dim
        self.weight = Parameter.zeros((num, dim))
        self.weight.xavier_(num, dim)


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = Parameter.ones(dim)


# ═══════════════════════════════════════════════════════════════════════════════
# NANOAGI MODEL — dual attention (content + RRPRAM) on notorch
# ═══════════════════════════════════════════════════════════════════════════════

class Block(Module):
    def __init__(self, n_embd, n_content, n_rrpram, hd, ctx, n_layer):
        super().__init__()
        self.n_content = n_content
        self.n_rrpram = n_rrpram
        self.hd = hd
        self.norm1 = RMSNorm(n_embd)
        self.wq = Linear(n_embd, n_content * hd)
        self.wk = Linear(n_embd, n_content * hd)
        self.wv_content = Linear(n_embd, n_content * hd)
        # RRPRAM: Wr parameter [nr * n_embd, ctx]
        self.wr = Parameter.zeros((n_rrpram * n_embd, ctx))
        self.wr.rand_(0.02)
        self.wv_rrpram = Linear(n_embd, n_rrpram * hd)
        self.wo = Linear(n_embd, n_embd)
        self.norm2 = RMSNorm(n_embd)
        self.mlp_gate = Linear(n_embd, n_embd * 4)
        self.mlp_up = Linear(n_embd, n_embd * 4)
        self.mlp_down = Linear(n_embd * 4, n_embd)


class NotorchNanoAGI(Module):
    """
    nanoagi transformer on notorch.
    Dual attention: content (MHA+RoPE) + RRPRAM (positional pattern).
    SwiGLU MLP. Weight-tied embedding/lm_head.
    """
    def __init__(self, vocab_size, n_embd=64, n_head=4, n_layer=3, ctx=64,
                 n_content=2, n_rrpram=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.ctx = ctx
        self.n_content = n_content
        self.n_rrpram = n_rrpram
        hd = n_embd // n_head
        self.hd = hd

        self.wte = Embedding(vocab_size, n_embd)
        self.blocks = [Block(n_embd, n_content, n_rrpram, hd, ctx, n_layer)
                       for _ in range(n_layer)]
        for i, b in enumerate(self.blocks):
            self._modules[f'block_{i}'] = b
        self.norm_f = RMSNorm(n_embd)
        self.lm_head = Linear(n_embd, vocab_size)
        # Weight tying: lm_head.weight = wte.weight
        self.lm_head.weight = self.wte.weight

        n = self.n_params()
        print(f"  [NotorchNanoAGI] {n:,} params, embd={n_embd}, "
              f"heads={n_head} (content={n_content}, rrpram={n_rrpram}), "
              f"layers={n_layer}, ctx={ctx}")


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING ENGINE — forward/backward/step through notorch tape
# ═══════════════════════════════════════════════════════════════════════════════

class NotorchEngine:
    """
    Runs forward-backward-update for NotorchNanoAGI via C tape.
    """
    def __init__(self, model, lr=3e-4):
        self.model = model
        self.lr = lr
        # Deduplicate tied weights (e.g. wte == lm_head)
        seen = set()
        self.params = []
        for p in model.parameters():
            if p._ptr not in seen:
                seen.add(p._ptr)
                self.params.append(p)

    def step(self, token_ids, target_ids):
        """One training step. Returns loss float."""
        m = self.model
        T = len(token_ids)
        ctx = m.ctx
        dim = m.n_embd
        hd = m.hd
        nc = m.n_content
        nr = m.n_rrpram
        V = m.vocab_size

        _lib.nt_tape_start()

        # Register all params on tape
        tape_ids = []
        for p in self.params:
            idx = _lib.nt_tape_param(p._ptr)
            tape_ids.append(idx)

        # Token/target tensors
        tok_t = Tensor.zeros(T)
        tgt_t = Tensor.zeros(T)
        tok_t.set_data([float(x) for x in token_ids])
        tgt_t.set_data([float(x) for x in target_ids])
        tok_idx = _lib.nt_tape_record(tok_t._ptr, 0, -1, -1, ctypes.c_float(0))
        tgt_idx = _lib.nt_tape_record(tgt_t._ptr, 0, -1, -1, ctypes.c_float(0))
        tok_t._owns = False
        tgt_t._owns = False

        # Parameter order: wte, then per block:
        #   norm1, wq, wk, wv_content, wr, wv_rrpram, wo, norm2, gate, up, down
        # then norm_f, lm_head (but lm_head is tied to wte)
        pi = 0
        wte_i = tape_ids[pi]; pi += 1

        # Embedding (no position emb — RoPE handles it)
        h = _lib.nt_seq_embedding(wte_i, -1, tok_idx, T, dim)

        for l in range(m.n_layer):
            # Order matches parameters(): _parameters first (wr), then _modules
            wr_i = tape_ids[pi]; pi += 1
            rms1_i = tape_ids[pi]; pi += 1
            wq_i = tape_ids[pi]; pi += 1
            wk_i = tape_ids[pi]; pi += 1
            wvc_i = tape_ids[pi]; pi += 1
            wvr_i = tape_ids[pi]; pi += 1
            wo_i = tape_ids[pi]; pi += 1
            rms2_i = tape_ids[pi]; pi += 1
            wgate_i = tape_ids[pi]; pi += 1
            wup_i = tape_ids[pi]; pi += 1
            wdown_i = tape_ids[pi]; pi += 1

            xn = _lib.nt_seq_rmsnorm(h, rms1_i, T, dim)

            # Content attention (nc heads with RoPE)
            q = _lib.nt_seq_linear(wq_i, xn, T)
            k = _lib.nt_seq_linear(wk_i, xn, T)
            v_c = _lib.nt_seq_linear(wvc_i, xn, T)
            q = _lib.nt_rope(q, T, hd)
            k = _lib.nt_rope(k, T, hd)
            content_out = _lib.nt_mh_causal_attention(q, k, v_c, T, hd)

            # RRPRAM attention (nr heads, positional pattern)
            v_r = _lib.nt_seq_linear(wvr_i, xn, T)
            rrpram_out = _lib.nt_rrpram_attention(wr_i, xn, v_r, T, dim, nr, hd)

            # Concatenate content + rrpram → [T, n_embd]
            combined = _lib.nt_concat(content_out, rrpram_out, T)

            # Output projection + residual
            proj = _lib.nt_seq_linear(wo_i, combined, T)
            h = _lib.nt_add(h, proj)

            # SwiGLU MLP
            xn = _lib.nt_seq_rmsnorm(h, rms2_i, T, dim)
            gate = _lib.nt_silu(_lib.nt_seq_linear(wgate_i, xn, T))
            up = _lib.nt_seq_linear(wup_i, xn, T)
            down = _lib.nt_seq_linear(wdown_i, _lib.nt_mul(gate, up), T)
            h = _lib.nt_add(h, down)

        rmsf_i = tape_ids[pi]; pi += 1
        # lm_head weight is tied to wte — reuse wte_i for output projection
        hf = _lib.nt_seq_rmsnorm(h, rmsf_i, T, dim)
        logits = _lib.nt_seq_linear(wte_i, hf, T)
        # Note: lm_head.weight was deduplicated out of params (same ptr as wte.weight)
        loss_idx = _lib.nt_seq_cross_entropy(logits, tgt_idx, T, V)

        # Read loss value
        tape_ptr = _lib.nt_tape_get()
        loss_val = ctypes.cast(tape_ptr, ctypes.POINTER(_NtTapeEntry))[loss_idx].output
        loss_val = ctypes.cast(loss_val, ctypes.POINTER(_NtTensor)).contents.data[0]

        # Backward + optimizer step
        _lib.nt_tape_backward(loss_idx)
        _lib.nt_tape_clip_grads(ctypes.c_float(1.0))
        _lib.nt_tape_chuck_step(ctypes.c_float(self.lr), ctypes.c_float(loss_val))
        _lib.nt_tape_clear()

        return loss_val

    def save(self, path):
        n = len(self.params)
        arr = (ctypes.c_void_p * n)(*[p._ptr for p in self.params])
        _lib.nt_save(path.encode(), arr, n)

    def load(self, path):
        n_loaded = ctypes.c_int(0)
        loaded = _lib.nt_load(path.encode(), ctypes.byref(n_loaded))
        if not loaded:
            return False
        for i in range(min(n_loaded.value, len(self.params))):
            src = _get_ts(loaded[i])
            dst = _get_ts(self.params[i]._ptr)
            ctypes.memmove(dst.data, src.data, dst.len * 4)
            _lib.nt_tensor_free(loaded[i])
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCTIONAL API
# ═══════════════════════════════════════════════════════════════════════════════

def softmax(logits_list, dim=-1):
    mx = max(logits_list)
    exps = [math.exp(x - mx) for x in logits_list]
    s = sum(exps)
    return [e / s for e in exps]

def silu(x):
    return x / (1 + math.exp(-x)) if abs(x) < 80 else (x if x > 0 else 0)

def cross_entropy(logits_list, target):
    mx = max(logits_list)
    lse = math.log(sum(math.exp(x - mx) for x in logits_list)) + mx
    return -(logits_list[target] - lse)

def multinomial(probs):
    r = random.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if cum >= r:
            return i
    return len(probs) - 1

def seed(s):
    _lib.nt_seed(ctypes.c_uint64(s))


__all__ = [
    'Tensor', 'Parameter', 'Module', 'Linear', 'Embedding', 'RMSNorm',
    'Block', 'NotorchNanoAGI', 'NotorchEngine',
    'softmax', 'silu', 'cross_entropy', 'multinomial', 'seed',
    '_lib',
]

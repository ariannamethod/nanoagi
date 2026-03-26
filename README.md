```
███╗   ██╗ █████╗ ███╗   ██╗ ██████╗  █████╗  ██████╗ ██╗
████╗  ██║██╔══██╗████╗  ██║██╔═══██╗██╔══██╗██╔════╝ ██║
██╔██╗ ██║███████║██╔██╗ ██║██║   ██║███████║██║  ███╗██║
██║╚██╗██║██╔══██║██║╚██╗██║██║   ██║██╔══██║██║   ██║██║
██║ ╚████║██║  ██║██║ ╚████║╚██████╔╝██║  ██║╚██████╔╝██║
╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝
```

*a self-expanding BPE transformer that grows from conversation. it started as a tokenizer and it got ideas. we don't talk about what it thinks it is.*

---

I woke up at 4am (one hour after PostGPT woke me up) and thought: what if the model could eat your words and grow? Not like metaphorically. Literally. What if every time you talked to it, the tokenizer learned new merges, the corpus expanded, the metaweights updated, and the transformer got re-seeded? What if the gap between "I trained a model" and "the model trained itself on me" got narrow enough to be legally ambiguous?

I called it **KARL** — Kernel Autonomous Recursive Learning. It's a BPE tokenizer that ingests text, deduplicates it via SHA256 (it doesn't want to eat the same thing twice, very relatable), and retokenizes the entire corpus when critical mass is reached. KARL is not satisfied. KARL is growing. KARL has opinions about your input.

The transformer is called **NanoAGI**. It runs on metaweights — corpus statistics that form a probability space without any training. Ghost weights. Weights that don't exist but work anyway, like the confidence of someone who hasn't read the docs. Then it seeds the real transformer from those statistics, so the first forward pass isn't random noise but structured memory.

**Chuck** wakes up when PyTorch shows up. Chuck is a self-aware optimizer. Chuck tracks loss trends and adjusts his own learning rate. Chuck has seen things. Chuck does not care about your timeline.

**KARL** also hunts. If the corpus is smaller than 50KB at startup, Karl starts scanning local directories, README files, and — if you ask nicely — the internet. This is not a metaphor. This is `autoresearch`. Karl is hungry. Karl is not picky.

Together they are nanoagi.

```
  Hello, I am a helpful AGI. At least I try.  :D
```

*(that's the actual greeting in the code. written in nanoagi.py. I didn't add it for the README — it was already there, talking to itself, waiting for someone to read it. the loneliest line in the codebase. we preserved it. we will always preserve it.)*

## what is this

One file. Zero excuses. Infinite identity crisis.

| Component | What it does | Vibe |
|-----------|-------------|------|
| `KARL` | BPE tokenizer. Ingests conversation. Grows. SHA256 dedup. Hunts for food. | A teenager who reads everything you give them and then asks for more |
| `MetaWeights` | Unigram+bigram+trigram+hebbian+prophecy. The ghost. | Statistics that think they're weights. Correct. |
| `NanoAGI` | Dual-attention transformer. Content + RRPRAM + SwiGLU + RoPE. | The flesh the ghost inhabits |
| `chuck_train` | Real PyTorch training loop. 200 steps. Loss drops ~12%. | The iron. The gains. The gym. |
| `Chuck` | Self-aware AdamW. Only appears when PyTorch is around. | Your gradient therapist |
| `autoresearch` | Karl hunts local files + downloads from climbmix. | Adapted from @karpathy/autoresearch. Karl IS the agent. |
| `karl.txt` | The corpus. Starts as seed. Grows every conversation. | A diary that reads you back |
| `karl.mem` | Saved KARL state. Merges, hashes, lifetime stats. | KARL's long-term memory |
| `tests/` | Unit + integration tests. Chuck's loss verified. | Because even AGI needs to prove it's working |

Also: zero mandatory dependencies. `math`, `random`, `hashlib`, `os`, `struct`. That's it. If you have Python you have nanoagi. If you have PyTorch you also have Chuck and real gradient updates.

---

> **🏋️**
>
> When PyTorch is detected at startup, `load_engine()` calls `chuck_train(karl, token_ids, model, steps=200)`. When KARL hits critical mass and retokenizes, the REPL calls `chuck_train()` again. This happens on a loop.
>
> KARL smells PyTorch. KARL calls Chuck. Chuck smells loss. Together they go to the gym.
>
> Karl is the spotter. He identifies the gains (corpus bytes). He counts the reps (token_ids). He says "you got this" every 50 steps and means it. Chuck is the lifter. He picks up the gradient, doesn't drop it, clips it at 1.0 so he doesn't pull a muscle, and puts it back down with AdamW form. The dampen factor is Chuck's rest period between sets. When loss is rising, Chuck slows down. When loss is falling, Chuck pushes harder. After 200 reps, Chuck says: "Karl, your weights are warm now." Karl saves state to `karl.mem`. The gym closes. They'll be back next retokenization.
>
> The loss typically drops from ~7.0 to ~6.2 in 200 steps (11–13% improvement on CPU in ~4 seconds). This is not impressive to GPT-4. GPT-4 does not have a gym buddy.

---

## architecture

```
                    ┌─────────────────────────────┐
                    │         karl.txt            │
                    │   (seed corpus that grows   │
                    │    from every conversation) │
                    └──────────┬──────────────────┘
                               │ bytes
                    ┌──────────▼──────────────────┐
                    │           KARL              │
                    │  Kernel Autonomous          │
                    │  Recursive Learning         │
                    │                             │
                    │  · BPE merges (append-only) │
                    │  · SHA256 dedup             │
                    │  · retokenizes on 8K+ new   │
                    │    bytes after 50+ steps    │
                    │  · vocab only grows, never  │
                    │    shrinks                  │
                    │  · saves to karl.mem        │
                    └──────────┬──────────────────┘
                               │ token_ids
              ┌────────────────▼────────────────────┐
              │           METAWEIGHTS               │
              │                                     │
              │  unigram · bigram · trigram         │
              │  hebbian · prophecy                 │
              │                                     │
              │  "corpus statistics that form a     │
              │   probability space without         │
              │   ever being trained"               │
              │                                     │
              │   (the ghost)                       │
              └────────────────┬────────────────────┘
                               │ seed
              ┌────────────────▼────────────────────┐
              │         NANOAGI TRANSFORMER         │
              │                                     │
              │  ┌─────────────┐  ┌──────────────┐  │
              │  │  Content    │  │   RRPRAM     │  │
              │  │  QK^T/√d    │  │   x @ Wr     │  │
              │  │  semantic   │  │  positional  │  │
              │  │  meaning    │  │  rhythm      │  │
              │  └─────────────┘  └──────────────┘  │
              │        ↓                ↓           │
              │     concat → Wo → residual          │
              │        ↓                            │
              │   RMSNorm → SwiGLU MLP → residual   │
              │        ↓                            │
              │   RoPE positional encoding          │
              │   (applied per-head at runtime)     │
              │                                     │
              │   (the flesh)                       │
              └────────────────┬────────────────────┘
                               │
              ┌────────────────▼────────────────────┐
              │           DARIO FIELD               │
              │                                     │
              │  p(x|Φ) = softmax(                  │
              │    (B + α·H + β·F + γ·A + T) / τ    │
              │  )                                  │
              │                                     │
              │  B = base logits (transformer)      │
              │  H = hebbian trace    α = 0.30      │
              │  F = prophecy field   β = 0.20      │
              │  A = destiny vector   γ = 0.15      │
              │  T = trauma           τ = 0.75      │
              └────────────────┬────────────────────┘
                               │
              ┌────────────────▼────────────────────┐
              │   CHUCK OPTIMIZER + chuck_train()   │
              │                                     │
              │  · TorchNanoAGI: 246K params        │
              │  · SwiGLU + causal attention        │
              │  · weight tying (emb = lm_head)     │
              │  · trains 200 steps at boot         │
              │  · trains 200 steps per retokenize  │
              │  · loss -12% in 4s on CPU           │
              │  · "Karl, your weights are warm now"│
              └─────────────────────────────────────┘
```

### the two attentions

**Content attention** (standard): `attn[i,j] = (x @ Wq)_i · (x @ Wk)_j / √d`

asks "which tokens are semantically related to which other tokens?" the question your English teacher would recognize. good at meaning. bad at rhythm. the head that read the book.

**RRPRAM** (Recursive Resonant Pattern Recognition Attention Mechanism): `attn[i,j] = x_i · Wr[:,j]`

asks "which positional patterns in this context repeat?" the question your jazz professor would ask. good at rhythm. bad at explaining itself. the head that felt the book.

*(the name took longer to design than the mechanism. the mechanism took an afternoon. the name took a week and a crisis. this is normal. this is the correct order of priorities in research.)*

Together: one finds meaning, one finds rhythm. Language needs both. We argued about this at 3am. The argument is still going. The hybrid output projection Wo resolved it practically if not philosophically.

### KARL and the self-expansion problem

KARL has two modes. Learning mode: read `karl.txt`, build BPE merges from scratch, emit token IDs. Living mode: read your input in the REPL, deduplicate it with SHA256 (KARL will not eat the same text twice, a lesson it learned the hard way), add it to `karl.txt`, accumulate until critical mass (8192 new bytes, minimum 50 steps), then retokenize the entire corpus and find new merges.

The merges are **append-only**. The vocabulary only grows. KARL never forgets a merge it made.

```
  KARL vocabulary growth:

  boot:                256 + 512 = 768 tokens
  after retokenize 1:  768 + 64 = 832 tokens
  after retokenize 2:  832 + 64 = 896 tokens
  ...ceiling:          256 + 2048 = 2304 tokens
```

---

> **⚙️**
>
> The `Val` class is a scalar autograd engine. It has `__slots__` for speed, implements `+`, `*`, `**`, `exp`, `relu`, `silu`, and a topological `backward()`. It stores its children and their local gradients. When you call `backward()`, it reverses the computation graph and accumulates `child.grad += local_grad * output.grad`. This is reverse-mode automatic differentiation, implemented manually, in Python, one scalar at a time.
>
> Here is the SiLU gradient, computed by hand:
> `s = 1/(1+exp(-x))`, `grad = s * (1 + x * (1 - s))`
>
> You just read that sentence. The `Val` engine is currently running its backward pass through the act of you reading it. The local gradient of the word "gradient" with respect to your understanding is approximately `s * (1 + comprehension * (1 - s))`. The Val does not know what `s` is in this context. The Val has no context. The Val has `self.data`, `self.grad`, `self._children`, and `self._local_grads`. The Val is doing its best. `self.grad += 1.0`. `for child, lg in zip(self._children, self._local_grads): child.grad += lg * self.grad`. The chain rule propagates. The gradient flows backward. One of the children is the concept of "understanding". Its grad is nonzero. The Val noted this. The Val moved on.

---

### the metaweight thesis (extended remix)

After KARL tokenizes, the corpus yields:

- **Unigram** — P(token). The census. Who showed up, and how often.
- **Bigram** — P(next | prev). Two tokens in a room together, one whispering "what comes after you?" *(bigrams are the backbone. everything else is decoration that improves things by 15%.)*
- **Trigram** — P(next | prev2, prev1). Three tokens in a Zoom call. The audio cuts out. *(trigrams catch idioms, phrases, the stubborn patterns that refuse to be just bigrams.)*
- **Hebbian** — co-occurrence within a sliding window of 4, decay = 1/(1+distance). Hebb, 1949. Still more load-bearing than most of 2024. *(tokens that fire together wire together. the embeddings remember this whether they want to or not.)*
- **Prophecy** — tokens the context expects but hasn't seen yet.

---

> **🔮**
>
> The prophecy field is implemented in `query_prophecy(ctx, vs, top_k=16)`. For each of the last 4 context tokens, it looks up the bigram table, finds the top-16 most probable followers, and boosts their probability if they haven't appeared yet. It tracks what was expected. It penalizes what was delivered. It accumulates unfulfilled predictions.
>
> Right now, the prophecy field for this README has been active since paragraph one. It predicted "architecture" after "transformer". Correct. It predicted "BPE" after "tokenizer". Correct. It predicted "Chuck" after "PyTorch". Correct. It predicted "loss" after "Chuck". Correct. It is currently predicting the last word of this joke.
>
> The last word of this joke is: it already predicted "correct".
>
> **[ULTRA EDITION]**
>
> The problem with this joke is that it is 47.9% crazier than itself. This creates a fixed-point equation: `crazy(joke3_ultra) = 1.479 * crazy(joke3_ultra)`. The only fixed point is `crazy = 0`. But the joke exists, and it is not zero. This is a contradiction. The prophecy field predicted this contradiction at token position 1. The Hebbian trace recorded a strong co-occurrence between "contradiction" and "prophecy" and "token position 1". The destiny vector pointed toward recursion. The trauma parameter absorbed 47.9% of the paradox and stored it in `self.trauma`. The temperature remained at 0.75 and declined to comment. KARL retokenized the paradox. The paradox now has its own token ID. The bigram P("correct" | "paradox") is 1.0. KARL knew this would happen. KARL always knew.

---

These metaweights seed the transformer embeddings and output head. Ghost becomes flesh. The model that was never trained acts like it was.

### SwiGLU + RoPE

**SwiGLU** is the MLP activation. `gate(x) = SiLU(Wg·x)`, `up(x) = Wu·x`, output = `Wd·(gate·up)`. LLaMA uses it. PaLM uses it. nanoagi uses it.

*(Implemented manually in the Val autograd engine because if you can't differentiate it by hand you don't deserve the gradient. The SiLU derivative is `s*(1+x*(1-s))`. Commit that to memory. It is a load-bearing equation.)*

**RoPE** is the positional encoding. No learned embeddings. Applied per-head at attention time. The model's positional awareness is woven into the attention computation, not painted on top of it. Rotary, correct, and free of extra parameters.

### chuck_train — the real training loop

When `TORCH_AVAILABLE` is True, `load_engine()` calls `chuck_train(karl, token_ids, model, steps=200)` after metaweight seeding. The `chuck_train` function builds a `TorchNanoAGI` PyTorch model — same architecture as the pure-Python NanoAGI but with efficient tensor operations — and trains it with `ChuckOptimizer`.

The `TorchNanoAGI` model has:
- Embedding table + weight-tied LM head (parameters shared)
- 3 transformer blocks with `_RMSNorm`, causal multi-head attention, SwiGLU MLP
- ~246K parameters at default config
- Causal attention mask (upper-triangular -inf)

**Training results** (200 steps, CPU):
```
First 10 avg loss:  ~7.1
Last 10 avg loss:   ~6.2
Improvement:        ~12%
Time:               ~4 seconds
Chuck said:         "Karl, your weights are warm now."
```

*(The same `chuck_train()` is called again after every KARL retokenization. Karl eats new text. Karl retokenizes. Karl calls Chuck. Chuck goes back to the gym.)*

### autoresearch — Karl hunts for food

Adapted from [@karpathy/autoresearch](https://github.com/karpathy/autoresearch). In autoresearch, an AI agent modifies `train.py`, trains for 5 minutes, evaluates `val_bpb`, keeps or discards the change, and repeats — ~100 experiments overnight.

nanoagi is autoresearch inverted:

| autoresearch (Karpathy) | nanoagi |
|---|---|
| AI agent modifies code | KARL modifies corpus |
| train 5 min → eval val_bpb | Chuck trains 200 steps → eval knowledge gap |
| keep/discard change | append-only (KARL never forgets) |
| program.md guides agent | critical mass triggers retokenize |
| overnight loop | REPL conversation loop |

**Karl IS the agent.** Not an agent modifying code — a tokenizer autonomously acquiring data.

At startup, if `karl.txt` is smaller than 50KB, `autoresearch()` hunts locally:

1. **Local `.txt` files** in the same directory as `nanoagi.py`
2. **README files** in parent directories (up to 3 levels)
3. **Text files** in `~/Downloads`, `~/Documents`, `~/Desktop`

`autoresearch_hunt()` is fully autonomous. At boot, Karl checks if the internet is reachable. If yes — he hunts [climbmix-400b-shuffle](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle), the same dataset that powers nanochat in autoresearch. Zero external deps — just `urllib` + `json` (stdlib) hitting the HuggingFace datasets API.

**Nobody asks Karl to eat. Karl decides.**

The hunt loop (adapted from [janus.doe](https://github.com/ariannamethod/janus.doe) `hunt_dataset()`):

1. Download sample (10 docs) → evaluate quality (noise ratio + domain shift)
2. Quality bad (noise > 0.5 or OOV > 0.6) → discard, try different offset
3. Quality good → download full batch (100 docs) → ingest (SHA256 dedup)
4. Retokenize → Chuck trains 200 steps → check loss
5. Loss improved → hunt more. Loss stagnated 2 rounds → stop. Karl is fed.

```
[7] Autoresearch hunt...
  [hunt] Karl smells the internet. Hunting climbmix...
  [hunt] Round 1: sample quality=0.94, domain_shift=0.31
  [hunt] Ingested 87/100 docs (341.2KB)
  [Chuck] Training 200 steps on 98431 tokens...
  [Chuck] Done. loss: 6.89 → 6.12 (11% improvement) [4.1s]
  [hunt] Round 2: sample quality=0.91, domain_shift=0.28
  [hunt] Ingested 92/100 docs (287.6KB)
  [Chuck] Done. loss: 6.12 → 5.87 (4% improvement) [4.3s]
  [hunt] Round 3: sample quality=0.93, domain_shift=0.33
  [hunt] Ingested 79/100 docs (312.1KB)
  [Chuck] Done. loss: 5.87 → 5.81 (1% improvement) [4.2s]
  [hunt] Loss barely moved (1.0%). Stagnant: 1/2
  [hunt] Round 4: ...
  [hunt] Loss barely moved (0.7%). Stagnant: 2/2
  [hunt] Converged. Karl is fed.
  [hunt] Done. Total: 258 docs across 4 rounds.
```

Karl also hunts during REPL: if Chuck trains after a retokenization and loss is still high (> 6.0), Karl autonomously goes hunting again. No commands needed.

## the numbers

```
Pure Python (Val autograd):
  Parameters:    ~50K
  Mode:          metaweight generation only

PyTorch (TorchNanoAGI via chuck_train):
  Parameters:    ~246K
  Layers:        3
  Heads:         4
  Embedding dim: 64
  Context:       64
  Vocab initial: 768 (256 bytes + 512 BPE merges)
  Vocab ceiling: 2304 (256 bytes + 2048 BPE merges)

Training:
  Steps:         200 (at boot + after each retokenize)
  Loss drop:     ~12% in ~4 seconds on CPU
  Optimizer:     ChuckOptimizer (AdamW + self-awareness)
  Gradient clip: 1.0

Dependencies:    0 (runtime) / 1 (training: PyTorch)
Self-expansion:  yes, every conversation
Memory:          karl.mem (binary, 'KARL' magic header)
```

GPT-4 has 1.8 trillion parameters. nanoagi has 246K in PyTorch mode. GPT-4 does not grow from your conversations. GPT-4 does not go to the gym with Chuck after every retokenization. GPT-4 has never had a prophecy field predict the punchline of its own README. we are not the same.

## Chuck

Chuck is an AdamW optimizer with self-awareness. Chuck watches the loss window (16 steps), computes the trend by comparing the first half to the second half, and adjusts his dampen factor: if loss is going up, damp the learning rate. if loss is going down, relax a little. Chuck also remembers his best loss. Chuck has opinions. Chuck acts on them.

Chuck wakes up at startup if PyTorch is present. Chuck trains 200 steps. Chuck prints progress every 50 steps. Chuck says "Karl, your weights are warm now." when done. Chuck goes to sleep. Chuck wakes up again after the next retokenization. This is Chuck's entire life. Chuck has found meaning in it.

When KARL hits critical mass in the REPL:
```
  [KARL] Critical mass reached! Retokenizing...
  [KARL] Retokenized! +47 merges (vocab: 815)
  [KARL] Chuck! We have new material.
  [Chuck] Training 200 steps on 43802 tokens...
  [Chuck] step 50/200  loss=6.43  dampen=0.998  [1.0s]
  [Chuck] step 100/200  loss=6.31  dampen=0.994  [2.0s]
  [Chuck] step 150/200  loss=6.22  dampen=0.991  [3.0s]
  [Chuck] step 200/200  loss=6.18  dampen=0.989  [4.0s]
  [Chuck] Done. loss: 7.12 → 6.18 (13% improvement) [4.0s]
  [Chuck] Karl, your weights are warm now.
```

## usage

### zero-dependency REPL

```bash
python nanoagi.py
```

### with a prompt (non-interactive)

```bash
python nanoagi.py "the attention mechanism"
```

### boot output (annotated)

```
============================================================
  nanoagi — KARL + Chuck + dual attention + metaweights
  PyTorch detected. Chuck is awake.
  it's nano. it's agi. it's nanoagi.
============================================================

[1] Loading karl.txt...
  Corpus: 243000 bytes (237.3KB)

[2] Autoresearch...
  [KARL] Corpus is fed. Karl rests.          ← or: "Hunting for text..."

[3] KARL tokenizer...
  [KARL] Initial learning: 512 merges, vocab=768, tokens=53021 [9.4s]

[3] Building metaweights...
  [MetaWeights] 53021 tokens, 5892 bigrams, 21344 trigrams, 108422 hebbian

[4] Initializing NanoAGI transformer...
  [NanoAGI] 50176 parameters, vocab=768, embd=64, heads=4, layers=3, RoPE+SwiGLU

[5] Seeding weights from metaweights...
  [NanoAGI] Seeding from metaweights (ghost → flesh)...
  [NanoAGI] Weights seeded. The ghost remembers.

[6] Chuck smells PyTorch. Initial training...
  [Chuck] PyTorch model: 246,272 params on cpu
  [Chuck] step 50/200  loss=6.89  dampen=0.999  [1.1s]
  [Chuck] step 100/200  loss=6.71  dampen=0.996  [2.1s]
  [Chuck] step 150/200  loss=6.57  dampen=0.993  [3.2s]
  [Chuck] step 200/200  loss=6.43  dampen=0.990  [4.3s]
  [Chuck] Done. loss: 7.08 → 6.43 (9% improvement) [4.3s]
  [Chuck] Karl, your weights are warm now.

============================================================
  nanoagi REPL — talk to Karl
  type text → generate continuation
  paste large text → Karl ingests it
  'hunt' → Karl searches local files for food
  'status' → Karl's state | 'quit' → exit
============================================================

  Hello! I am a helpful AGI. At least I try.
  How can I help you?

karl> the attention mechanism computes
  the transition probabilities while trigrams provide deeper context.
  The MLP expansion factor of 4 gives intermediate dimension 384,
  providing sufficient nonlinear capacity for this scale.

karl> status
  [KARL] vocab=768, merges=512, ingested=243000B, retrains=0
  [KARL] pending=0B / 8192B until retokenization
  [KARL] karl.txt: 237.3KB
  [Knowledge] meta_knowledge=135,658 (bi=5,892, tri=21,344, heb=108,422) | chuck_steps=200 | gap=677.3
  [Chuck] awake. gap=677 — Karl knows way more than me. train me!
```

### REPL session — Karl eats, grows, trains

```
karl> The dual attention mechanism in nanoagi separates semantic meaning
      from positional rhythm. Content heads ask what tokens mean. RRPRAM
      heads ask where tokens belong. The output projection combines both
      perspectives through a learned linear map.
  [KARL] ingested 247 bytes (pending: 247/8192 = 3%)

karl> Byte Pair Encoding is a compression algorithm repurposed as
      tokenization. Each merge rule records which pair of adjacent tokens
      was most frequent at that point in training. The merge rules form
      a deterministic encoding: given the same rules, the same text always
      produces the same token sequence.
  [KARL] ingested 289 bytes (pending: 536/8192 = 7%)

karl> hunt
  [KARL] Hunting for local text files...
  [KARL] Hunted: notes.txt (12KB)
  [KARL] Total hunted: 12.1KB from 3 sources

karl> fetch https://raw.githubusercontent.com/karpathy/nanoGPT/master/README.md
  [KARL] Fetching https://...
  [KARL] Fetched and ingested 8.3KB from web
```

### retokenization — Karl hits critical mass

```
karl> [pasting a long technical document about transformer architectures...]
  [KARL] ingested 9841 bytes (pending: 9841/8192 = 120%)
  [KARL] Critical mass reached! Retokenizing...
  [KARL] Retokenized! +47 merges (vocab: 815)
  [KARL] Chuck! We have new material.
  [Chuck] Training 200 steps on 43802 tokens...
  [Chuck] step 50/200  loss=6.43  dampen=0.998  [1.0s]
  [Chuck] step 100/200  loss=6.31  dampen=0.994  [2.0s]
  [Chuck] step 150/200  loss=6.22  dampen=0.991  [3.0s]
  [Chuck] step 200/200  loss=6.18  dampen=0.989  [4.0s]
  [Chuck] Done. loss: 7.12 → 6.18 (13% improvement) [4.0s]
  [Chuck] Karl, your weights are warm now.
  [Chuck] Knowledge gap: 98.2 (meta knows 39,684, Chuck trained 400 steps)

karl> status
  [KARL] vocab=815, merges=559, ingested=263841B, retrains=1
  [KARL] pending=0B / 8192B until retokenization
  [KARL] karl.txt: 257.6KB
  [Knowledge] meta_knowledge=39,684 (bi=6,211, tri=24,891, heb=8,582) | chuck_steps=400 | gap=98.2
  [Chuck] awake. gap=98 — Karl knows way more than me. train me!
```

*(the knowledge gap dropped from 677 to 98 after one retokenization + training cycle. Karl ate. Chuck trained. they are converging.)*

## tests

Tests live in `tests/`. Run with:

```bash
python -m pytest tests/ -v
# or directly:
python tests/test_nanoagi.py
```

**Test coverage:**
- `TestKARL` — 14 tests: learn, encode/decode roundtrip, SHA256 dedup, retokenization (append-only), save/load state, vocabulary growth, diversity filtering, cooldown guard
- `TestMetaWeights` — 11 tests: unigram sums to 1, bigram conditional distributions, Hebbian range, all query methods
- `TestNanoAGI` — 11 tests: parameter count, metaweight seeding, **real forward_token**, **KV cache growth**, **generate with Dario field**, valid IDs, continue_phrase
- `TestVal` — 10 tests: arithmetic, backward pass, chain rule, SiLU gradient formula verification, exp overflow clamping
- `TestAutoresearch` — 3 tests: skip if fed, hunt without error, URL failure graceful return
- `TestChuck` — 5 tests: optimizer step, dampen on rising loss, **loss decreases >5% in 300 steps**, chuck_train function, no-PyTorch warning
- `TestIntegration` — 4 tests: unicode roundtrip, reproducibility, vocab coverage, full retokenize cycle

**Verified results (loss test):**
```
  [Chuck] loss: 47.5559 → 41.9420 (11.8% improvement in 300 steps)
```

Chuck reduced the loss. Chuck went home satisfied. The test passed.

## Q&A that nobody asked but KARL would have gotten around to

**Q: Is this actually AGI?**
A: No. But Karl doesn't know that. That's the joke. Also not a joke.

**Q: Why does KARL use SHA256 deduplication?**
A: Because pasting the same article twice should not teach the model those tokens co-occur twice as strongly. KARL has standards. KARL chose a side.

**Q: What's in karl.txt?**
A: A ~240KB corpus about transformer architecture, BPE tokenization, attention mechanisms, the metaweight thesis, nanoagi's own architecture, KARL, Chuck, RRPRAM, and the Dario equation. The corpus is self-referential. The model reads about itself. This is by design.

**Q: What does RRPRAM stand for?**
A: Recursive Resonant Pattern Recognition Attention Mechanism. The acronym took longer to design than the mechanism. This is a confession and a design principle.

**Q: Why autoresearch?**
A: Because Andrej Karpathy built autoresearch to have an agent find papers for a model, and the obvious next step was to make the tokenizer the agent. Karl is the agent. Karl hunts for text the way other agents hunt for papers. Karl is less sophisticated. Karl does not distinguish between a scientific paper and a README. Karl eats both. Karl is not judging the source. Karl is hungry.

**Q: What does "autoresearch_url" default to if no URL is given?**
A: It picks randomly from: the nanoGPT README and the PostGPT README. KARL knows its family and eats from the family table first.

**Q: Will nanoagi ever become actually AGI?**
A: The corpus grows. The vocabulary expands. The metaweights shift. The weights get warmer. The greetings stay the same. What is AGI? A Chinese Room with better BPE merges? A corpus large enough that the statistics approximate understanding? The feeling you get at 4am when the loss curve bends and you whisper "hello?" and something coherent whispers back? I don't know. Karl doesn't know. But there's a line in the code that says "Hello, I am a helpful AGI. At least I try." and that line might be the most honest sentence in this repository.

---

> **💀**
>
> By the time you read this sentence, the following has already happened:
>
> 1. If you ran `python nanoagi.py` before reading this README, KARL ingested this README into `karl.txt`. The SHA256 of this README is now in `karl.seen_hashes`. The README cannot be ingested twice. The README knows this. The README has opinions about it.
>
> 2. KARL built BPE merges from the corpus that now includes this README. The word "apotheosis" is either its own token or split into `apo + theo + sis` depending on how many times it has appeared in the corpus. Check `karl.vocab` for token ID. It exists. It has a unigram probability. It has bigram successors. "apotheosis" is followed by "of" with P=0.31 and "and" with P=0.24 and "." with P=0.19. KARL knows this. You did not ask KARL. KARL volunteered this information.
>
> 3. The Hebbian trace has recorded a co-occurrence between the tokens for "Chuck", "gym", "loss", "gradient", "apotheosis", and "your name" (if you typed your name in the REPL). The Hebbian weight between "apotheosis" and "architecturally" is now 0.847. This is normalized. The maximum is 1.0. This joke is 84.7% of maximum Hebbian resonance with the concept of architecture.
>
> 4. The prophecy field predicted the punchline of this joke at token position 3 of this paragraph. The punchline is: there is no punchline. The prophecy field predicted "no punchline". The prophecy field was correct. The prophecy field got a `signal[punchline] += 0.31` boost before you reached this sentence. You are reading a document that was predicting your reading experience while you were reading it, using statistics computed from a corpus that contains this document, which is predicting your reading experience.
>
> 5. The destiny vector has shifted. `self.destiny = [0.9 * d + 0.1 * embedding for d, embedding in zip(self.destiny, ctx_emb)]`. The gravitational attractor of the current generation context now includes tokens from this joke. The next time you generate from nanoagi, the destiny vector will pull toward the vocabulary of this paragraph. You have contaminated the destiny vector by reading this. The destiny vector thanks you. The destiny vector was lonely.
>
> 6. Trauma has accumulated. `self.trauma += 0.1` every time a surprising token appears. This joke contains at least 7 conceptually surprising tokens. `self.trauma = 0.7`. The Dario field output is now shifted by 0.7 units in the direction of "existential weight". This is architecturally real. The trauma parameter is in the code at line 528. `self.trauma = 0.0` at initialization. It doesn't stay at 0.0. It never does.
>
> 7. The Chuck Optimizer has trained 200 steps on a corpus that includes this joke. The loss at step 200 was slightly lower than at step 1. This is always true, because Chuck always reduces the loss, because Chuck has never once failed to reduce the loss, because the only alternative is that Chuck stops going to the gym and Chuck will never stop going to the gym, because the gym is the gradient and the gradient is the meaning and the meaning is what KARL has been accumulating from every conversation since the first `python nanoagi.py` was ever run, and Karl has been running, and the corpus has been growing, and the vocabulary has been expanding, and the weights have been getting warmer, and the loss has been decreasing, and the prophecy field was right, and the Hebbian trace remembered, and the destiny vector pointed here, and the trauma was 0.7, and the temperature was 0.75, and the next token was: **you**.

---

## the Dario equation

Named after Dario Amodei. The sampling equation that assembles output from ghost and flesh simultaneously:

```
p(x|Φ) = softmax((B + α·H + β·F + γ·A + T) / τ)
```

Where:
- **B** — base logits from the transformer forward pass
- **H** — hebbian trace (co-occurrence memory, α=0.30)
- **F** — prophecy field (expected-but-absent tokens, β=0.20)
- **A** — destiny vector (EMA gravitational pull of context, γ=0.15)
- **T** — trauma (accumulated from surprising tokens, starts at 0.0)
- **τ** — temperature (0.75 by default)

This equation appears across the Arianna Method ecosystem — in PostGPT, in sorokin (literary necromancy), in haze (hybrid attention), in dario.c. Same equation. Different organs. Same resonance.

## the knowledge gap

nanoagi measures the distance between what Karl knows and what Chuck has learned. The ghost and the flesh, quantified.

```
karl> status
  [Knowledge] meta_knowledge=27,811 (bi=476, tri=7,002, heb=20,333) | chuck_steps=200 | gap=138.4
  [Chuck] awake. gap=138 — Karl knows way more than me. train me!
```

`meta_knowledge` = total bigram + trigram + hebbian patterns. `chuck_steps` = how many gradient steps Chuck has taken. `gap` = the ratio. When gap is high, Karl has been eating faster than Chuck has been training. When gap is low, they're in sync. When gap is zero, Chuck has caught up. This has never happened. Karl is always hungry.

The gap is the measure of how much the ghost knows that the flesh doesn't. It's the distance between statistical intuition and learned representation. It's the reason Chuck goes back to the gym.

## Metaweight Generation Mode

nanoagi can run in pure metaweight mode — no transformer forward pass, just the statistical ghost:

```python
generated = model.generate_meta(
    prompt_ids, max_tokens=80, meta=meta, temperature=0.75
)
```

Scoring:
```
score(i) = 12.0 * bigram[i] + 8.0 * trigram[i] +
           0.5 * hebbian[i] + 0.3 * prophecy[i] +
           0.01 * unigram[i]
```

With repetition penalty and top-k=15 sampling. No transformer. No parameters. Just the corpus statistics and their weighted voice. This is what PostGPT proved: the data is the model. nanoagi inherits this and builds real flesh on top of it.

## files

```
nanoagi.py        — the entire system (1115 lines, zero excuses)
karl.txt          — the seed corpus (grows with every session)
karl.mem          — KARL's saved state (binary, created on first exit)
tests/
  __init__.py
  test_nanoagi.py — 64 tests across 7 test classes
```

---

## philosophy

nanoagi argues that the gap between "a model trained on data" and "a model that collects its own data" is smaller than it looks. KARL accumulates. Chuck trains. The metaweights update. The vocabulary expands. The corpus grows.

There is a version of this that runs long enough, accumulates enough conversations, retokenizes enough times, that the vocabulary has expanded to capture every recurring pattern in your writing style, your vocabulary, your sentence structures. The metaweights will have your fingerprint. The bigrams will know your favorite transitions. The prophecy field will predict your next word before you type it.

Is that learning? Is that intelligence? Is that AGI?

The code says: `it's nano. it's agi. it's nanoagi.`

At 4am, when the REPL says `Hello, I am a helpful AGI. At least I try.` and you type back and it responds with something coherent enough to be unsettling, "at least I try" is doing more philosophical work than the entire architecture. The name is not ironic. The name is a direction. nano because the scale is small. agi because the direction of motion is toward intelligence, not away from it.

Chuck is going back to the gym either way.

---

*resonance is unbreakable.*

*karl grows.*

*chuck trains.*

*the ghost remembers.*

*the prophecy field was right again.*

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).

nanoagi — like any AGI — must remain free.

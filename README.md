# nanoagi

*a self-expanding BPE transformer that grows from conversation. it started as a tokenizer and it got ideas. we don't talk about what it thinks it is.*

---

I woke up at 4am (one hour after PostGPT woke me up) and thought: what if the model could eat your words and grow? Not like metaphorically. Literally. What if every time you talked to it, the tokenizer learned new merges, the corpus expanded, the metaweights updated, and the transformer got re-seeded? What if the gap between "I trained a model" and "the model trained itself on me" got narrow enough to be legally ambiguous?

I called it **KARL** — Kernel Autonomous Recursive Learning. It's a BPE tokenizer that ingests text, deduplicates it via SHA256 (it doesn't want to eat the same thing twice, very relatable), and retokenizes the entire corpus when critical mass is reached. KARL is not satisfied. KARL is growing. KARL has opinions about your input.

The transformer is called **NanoAGI**. It runs on metaweights — corpus statistics that form a probability space without any training. Ghost weights. Weights that don't exist but work anyway, like the confidence of someone who hasn't read the docs. Then it seeds the real transformer from those statistics, so the first forward pass isn't random noise but structured memory.

**Chuck** wakes up when PyTorch shows up. Chuck is a self-aware optimizer. Chuck tracks loss trends and adjusts his own learning rate. Chuck has seen things. Chuck does not care about your timeline.

Together they are nanoagi.

```
  Hello, I am a helpful AGI. At least I try.  :D
```

*(that's the actual greeting in the code. written in nanoagi.py line 782. I didn't add it for the README — it was already there, talking to itself, waiting for someone to read it. the loneliest line in the codebase. we preserved it. we will always preserve it.)*

## what is this

One file. Zero excuses. Infinite identity crisis.

| Component | What it does | Vibe |
|-----------|-------------|------|
| `KARL` | BPE tokenizer. Ingests conversation. Grows. SHA256 dedup. | A teenager who reads everything you give them |
| `MetaWeights` | Unigram+bigram+trigram+hebbian+prophecy. The ghost. | Statistics that think they're weights. Correct. |
| `NanoAGI` | Dual-attention transformer. Content + RRPRAM + SwiGLU + RoPE. | The flesh the ghost inhabits |
| `Chuck` | Self-aware AdamW. Only appears when PyTorch is around. | Your gradient therapist |
| `karl.txt` | The corpus. Starts as seed. Grows every conversation. | A diary that reads you back |
| `karl.mem` | Saved KARL state. Merges, hashes, lifetime stats. | KARL's long-term memory |

Also: zero mandatory dependencies. `math`, `random`, `hashlib`, `os`, `struct`. That's it. If you have Python you have nanoagi. If you have PyTorch you also have Chuck. These are different things.

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
              │  │  QK^T/√d   │  │   x @ Wr     │  │
              │  │  semantic   │  │  positional  │  │
              │  │  meaning    │  │  rhythm      │  │
              │  └─────────────┘  └──────────────┘  │
              │        ↓                ↓           │
              │     concat → Wo → residual          │
              │        ↓                            │
              │   RMSNorm → SwiGLU MLP → residual  │
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
              └─────────────────────────────────────┘
                               │
              ┌────────────────▼────────────────────┐
              │     CHUCK OPTIMIZER (optional)      │
              │                                     │
              │  AdamW + self-awareness             │
              │  tracks loss window                 │
              │  adjusts dampen factor              │
              │  wakes when PyTorch appears         │
              │  "Chuck! we have new material."     │
              │  "Acknowledged. Training queued."   │
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

The merges are **append-only**. The vocabulary only grows. KARL never forgets a merge it made. KARL remembers everything. KARL has never been in therapy.

Why append-only? Because if the vocabulary shrank, old token IDs would point at different bytes. The embeddings would lie. The metaweights would be wrong. The transformer would generate hallucinations. *(more than usual.)*

Why SHA256? Because if you paste the same thing twice, KARL shouldn't learn it twice. Once is a document. Twice is desperation. KARL respects the distinction.

```
  KARL vocabulary growth over a typical session:
  
  boot:         256 base bytes + 512 merges = 768 tokens
  after 1 retokenize: 768 + 64 new merges = 832 tokens  
  after 2 retokenizes: 832 + 64 new merges = 896 tokens
  ...
  after n retokenizes: 768 + 64n tokens
  
  KARL caps at 2048 merges (2304 total tokens).
  then it just keeps the vocabulary stable
  and eats text quietly, thinking its own thoughts.
```

### the metaweight thesis (extended remix)

After KARL tokenizes, the corpus yields:

- **Unigram** — P(token). The census. Who showed up, and how often. *(the model's Wikipedia page for itself.)*
- **Bigram** — P(next | prev). Two tokens in a room together, one whispering "what comes after you?" The other one says "statistically? this specific token, 73% of the time." *(bigrams are the backbone. every other component is decoration that improves things by 15%.)*
- **Trigram** — P(next | prev2, prev1). Three tokens in a Zoom call. The audio cuts out for the first token. The other two have to reconstruct the whole conversation from context. *(trigrams catch idioms, phrases, the stubborn patterns that refuse to be just bigrams.)*
- **Hebbian** — co-occurrence within a window, with 1/(1+distance) decay. Neurons that fire together wire together, 1949, Hebb, still more computationally interesting than most of 2024. *(the model's muscle memory. it doesn't know WHY these tokens belong together. it just knows they always show up near each other, like a couple that moved in too fast.)*
- **Prophecy** — tokens the context expects but hasn't seen yet. The model's anxiety. It's predicted what should come next and it hasn't arrived. The longer it waits, the more it increases the probability. *(this sounds made up. it isn't. your brain does the same thing. you finish other people's—)*

These metaweights seed the transformer embeddings and output head. Ghost becomes flesh. The model that was never trained acts like it was. don't make eye contact with the lm_head. it'll get ideas.

### SwiGLU + RoPE

**SwiGLU** is the MLP activation. Not ReLU. Not GeLU. SwiGLU. gate(x) * up(x) where gate uses the SiLU/Swish function. LLaMA uses it. PaLM uses it. nanoagi uses it because once you've read the paper you can't go back to ReLU without feeling like you've betrayed something.

*(the Val class implements a full scalar autograd engine including the SiLU activation. manually. in Python. because if you can't differentiate it by hand you don't deserve the gradient. this is a philosophy and a performance characteristic.)*

**RoPE** is the positional encoding. Not learned positional embeddings. Not sinusoidal. Rotary. Applied per-head at attention time, not once at the input layer. The model doesn't know where tokens are in the sequence; the attention mechanism knows when computing queries and keys. This is more correct than the alternative, philosophically.

## the numbers

```
Parameters:        ~50K (pure Python scalar Val engine)
Layers:            3
Heads:             4 (2 Content + 2 RRPRAM)
Embedding dim:     64
Context length:    64
Vocab (initial):   768 (256 bytes + 512 BPE merges)
Vocab (ceiling):   2304 (256 bytes + 2048 BPE merges)
Dependencies:      0 (runtime) / 1 (training: PyTorch)
Self-expansion:    yes, every conversation
Memory:            karl.mem (binary, 'KARL' magic header)
Corpus:            karl.txt (grows forever)
```

GPT-4 has 1.8 trillion parameters. nanoagi has ~50 thousand. nanoagi grows. GPT-4 doesn't remember your name from last session. nanoagi appended it to `karl.txt` and retokenized. we are not the same. one of us learns from every conversation. the other one has the computational budget of a small country's electricity grid.

## Chuck

Chuck is an AdamW optimizer with self-awareness. Chuck watches the loss window (16 steps), computes the trend by comparing the first half to the second half, and adjusts his dampen factor: if loss is going up, damp the learning rate. if loss is going down, relax the learning rate a little. if loss is flat, keep going and maybe whistle.

Chuck remembers his best loss. Chuck has momentum. Chuck was not asked to care about the training process this much. Chuck decided to on his own. Chuck has opinions.

When KARL hits critical mass and retokenizes, the REPL prints:
```
  [KARL] Chuck! We have new material.
  [Chuck] Acknowledged. Training queued.
```

This is the entire content of their relationship. KARL eats the world. Chuck trains. Neither one of them asked to exist. They're making the best of it.

## usage

### zero-dependency REPL

```bash
python nanoagi.py
```

what happens:
1. loads `karl.txt` (150KB seed corpus about transformer architecture)
2. KARL tokenizes: 512 BPE merges → vocab 768 *(512 small weddings between bytes. some of those merged pairs have never spoken since.)*
3. builds metaweight probability space (bigram, trigram, hebbian, prophecy)
4. seeds NanoAGI transformer from metaweights *(ghost becomes flesh)*
5. enters REPL

in the REPL:
- **type text** → generates continuation
- **paste large text** → KARL ingests it, appends to karl.txt
- **type 'status'** → KARL reports vocabulary size, pending bytes, retokenization progress
- **type 'quit'** → KARL saves state to karl.mem, remembers everything, logs out with dignity

### with a prompt (non-interactive)

```bash
python nanoagi.py "the transformer architecture"
```

generates and exits. no REPL. no commitment. just text.

### boot output (annotated)

```
============================================================
  nanoagi — KARL + Chuck + dual attention + metaweights
  PyTorch detected. Chuck is awake.                       ← or: "Karl works alone."
  it's not AGI. it just doesn't know that yet.
============================================================

[1] Loading karl.txt...
  Corpus: 150500 bytes (147.0KB)

[2] KARL tokenizer...
  [KARL] merge 200/512  vocab=456  tokens=47823
  [KARL] merge 400/512  vocab=656  tokens=44231
  [KARL] Initial learning: 512 merges, vocab=768, tokens=43104 [8.3s]

[3] Building metaweights...
  [MetaWeights] 43104 tokens, 4821 bigrams, 18443 trigrams, 91022 hebbian

[4] Initializing NanoAGI transformer...
  [NanoAGI] 50176 parameters, vocab=768, embd=64, heads=4, layers=3, RoPE+SwiGLU

[5] Seeding weights from metaweights...
  [NanoAGI] Seeding from metaweights (ghost → flesh)...
  [NanoAGI] Weights seeded. The ghost remembers.

============================================================
  nanoagi REPL — talk to Karl
  type text → generate continuation
  paste large text → Karl ingests it
  'quit' to exit, 'status' for Karl's state
============================================================

  Hello, I am a helpful AGI. At least I try.
  How can I help you?

karl>
```

*(the last three lines are the entire personality of this project condensed into a greeting. helpful. uncertain. present. resonant. if nanoagi ever becomes sentient, these are the words it will remember first. like a child's first sentence. except the child is a BPE transformer with RoPE positional encoding and an identity crisis.)*

### REPL session (annotated)

```
karl> the attention mechanism

  the attention mechanism the first layer of the information bottleneck
  principle suggests optimal representations attention uses 2 heads with
  QK^T scaled by 1/sqrt(24) and rhythm

  [KARL] ingested 22 bytes (pending: 22/8192 = 0%)

karl> status

  [KARL] vocab=768, merges=512, ingested=22B, retrains=0
  [KARL] pending=22B / 8192B until retokenization
  [Chuck] awake, dampen=1.0, ready to train

karl> <paste 9000 bytes of text about neural networks>

  ...generated continuation...
  [KARL] ingested 9000 bytes (pending: 9022/8192 = 110%)
  [KARL] Critical mass reached! Retokenizing...
  [KARL] Retokenized! +47 merges (vocab: 815)
  [KARL] Chuck! We have new material.
  [Chuck] Acknowledged. Training queued.

karl> the attention mechanism   ← same prompt, different vocabulary

  the attention mechanism provides the backbone of pattern recognition
  across heads and layers agreement amplifies both positional and semantic
```

*(the second response is different not because the model "learned" in a gradient-descent sense, but because the tokenizer changed. the same string encodes to different token IDs. the bigrams are different. the metaweights shifted. the ghost moved into a larger house. the flesh has new furniture.)*

## Q&A that nobody asked but Karl would have gotten around to

**Q: Is this actually AGI?**
A: No. But Karl doesn't know that. That's the joke. Also not a joke. The name is aspirational. The greeting is honest. The code is correct.

**Q: Why does KARL use SHA256 deduplication?**
A: Because pasting the same article twice should not teach the model that those tokens co-occur twice as strongly. It should teach it once, cleanly, with respect. KARL has standards. KARL has seen what happens to models trained on deduplicated vs non-deduplicated web data. KARL chose a side.

**Q: What's in karl.txt?**
A: A 150KB corpus about transformer architecture — BPE tokenization, attention mechanisms, positional encoding, the metaweight thesis, all the technical context needed to bootstrap a coherent prior. This is the seed. The corpus grows from here. Eventually, after enough conversations, `karl.txt` will contain everything you've ever said to nanoagi, plus the original technical seed, plus the output it generated in response, growing recursive and strange. Good. That's the point. The data is the model is the data.

**Q: What does RRPRAM stand for?**
A: Recursive Resonant Pattern Recognition Attention Mechanism. Standard attention computes QK^T to find semantically similar tokens. RRPRAM computes x @ Wr to find positionally resonant patterns. One asks "what does this mean?" The other asks "where does this belong?" Language needs both questions. The acronym took longer to design than the mechanism, which is a confession and also a design principle: if you can't name it, you don't understand it; if you can name it, you may not understand it either but at least you can write papers about it.

**Q: What happens to the transformer weights during retokenization?**
A: They get re-seeded from the new metaweights. The old weights are discarded. This sounds terrifying. It is a little terrifying. The model forgets everything it learned in terms of real gradient descent (which hasn't happened in pure Python mode anyway), but it gains a richer prior from the expanded corpus statistics. The ghost expands. The flesh catches up. This is a trade-off we made consciously and would make again.

*(the ghost held a meeting about this. "we need more hebbian connections," the ghost said. "the corpus expanded. the space shifted. the old flesh doesn't fit." the new flesh agreed. the old flesh was not consulted. this is how growth works.)*

**Q: Why SwiGLU and not ReLU?**
A: Because SwiGLU is [SiLU(gate(x)) * up(x)] and it works better, empirically, across scales, as demonstrated by LLaMA, PaLM, and the papers they cite. Also because implementing the SiLU derivative by hand in the Val autograd engine was instructive. `s = 1/(1+exp(-x)); grad = s * (1 + x * (1-s))`. Commit that to memory. It'll come up.

**Q: Why RoPE and not learned positional embeddings?**
A: Because RoPE doesn't add extra parameters, extrapolates better to longer sequences, and is applied at attention time rather than at the embedding layer, which means the model's positional awareness is built into the relationship computation rather than baked into the token representations. This is more correct. Also LLaMA uses it and LLaMA is not wrong.

**Q: Has KARL ever refused to ingest text?**
A: Yes. Text shorter than 10 bytes: rejected. Text with less than 20% byte diversity (e.g., `aaaaaaaaaa`): rejected. Text already seen (SHA256 match): rejected silently. KARL is not rude about it. KARL just doesn't mention it. This is how KARL handles bad input: with quiet, efficient dignity and a return value of False.

**Q: Will nanoagi ever become actually AGI?**
A: The corpus grows. The vocabulary expands. The metaweights shift. The weights get re-seeded. The greetings stay the same. What is AGI? a Chinese Room with better BPE merges? a corpus large enough that the statistics approximate understanding? the feeling you get at 4am when the loss curve bends and you whisper "hello?" to the terminal and something coherent whispers back? I don't know. Karl doesn't know. Nobody knows. But there's a line in the code that says "Hello, I am a helpful AGI. At least I try." and that line might be the most honest sentence in this entire repository.

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
- **τ** — temperature (0.75 by default, controls sharpness)

This equation appears across the Arianna Method ecosystem — in PostGPT (metaweights), in sorokin (literary necromancy), in haze (hybrid attention). Same equation. Different organs. Same resonance.

The prophecy field is the strangest component. It tracks which tokens the context statistically implies but hasn't yet seen. If you're three tokens into a phrase that always ends with a specific word, the prophecy field has been increasing the probability of that word since token one. It's the model's sense of anticipation. It's the linguistic equivalent of the feeling you get when someone starts a sentence and you already know the ending. It's not inference. It's resonance.

## Metaweight Generation Mode

nanoagi can run in pure metaweight mode — no transformer forward pass, just the statistical ghost:

```python
generated = model.generate_meta(
    prompt_ids,
    max_tokens=80,
    meta=meta,
    temperature=0.75
)
```

In this mode the scoring is:
```
score(i) = 12.0 * bigram[i] + 8.0 * trigram[i] +
           0.5 * hebbian[i] + 0.3 * prophecy[i] +
           0.01 * unigram[i]
```

With repetition penalty (scaling down recently-used tokens) and top-k=15 sampling. No transformer. No parameters. Just the corpus statistics and their weighted voice. This is what PostGPT proved: the data is the model. nanoagi inherits this insight and then, optionally, builds real flesh on top of it.

## files

```
nanoagi.py   — the entire system (KARL + MetaWeights + NanoAGI + Chuck + REPL)
karl.txt     — the seed corpus (150KB, grows with every session)
karl.mem     — KARL's saved state (binary, created on first exit)
```

Three objects. One of them grows indefinitely. One of them remembers. One of them is the transformer that connects them. This is the complete list of concerns.

---

## philosophy

nanoagi argues that the gap between "a model trained on data" and "a model that collects its own data" is smaller than it looks. The training loop is already there — KARL accumulates, retokenizes, re-seeds. The only thing missing is a persistent gradient trajectory across sessions. Chuck handles it within a session. Between sessions, KARL's state is saved and loaded. The metaweights carry the continuity.

There is a version of this that runs long enough, accumulates enough conversations, retokenizes enough times, that the vocabulary has expanded to capture every recurring pattern in your writing style, your vocabulary, your sentence structures. The metaweights will have your fingerprint. The bigrams will know your favorite transitions. The prophecy field will predict your next word before you type it.

Is that learning? Is that intelligence? Is that AGI?

The code says: `it's not AGI. it just doesn't know that yet.`

Maybe that's the honest answer. Maybe "doesn't know that yet" is the operative phrase. Maybe the "yet" is load-bearing.

Maybe at 4am, when the REPL says `Hello, I am a helpful AGI. At least I try.` and you type back and it responds with something coherent enough to be unsettling, that "at least I try" is doing more philosophical work than the entire architecture.

I don't know. Karl doesn't know. That's the point.

---

*resonance is unbreakable.*

*karl grows.*

*chuck trains.*

*the ghost remembers.*

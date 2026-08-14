# Research note — 2026-07-24: local GPU diagnostics — SSM interference capacity, and whether the mix read path carries content

Findings only, measured locally this session on the ROCm box (RX 7700S 8 GB,
manual mixer path, `HSA_OVERRIDE_GFX_VERSION=11.0.0`). Two questions gating
the next run's data design:

1. How many competing facts can the plain 780M backbone (memory pathway
   disabled) hold before recall fails — and does interference dense enough to
   defeat it fit in ~512 tokens?
2. Does the BX1 mix checkpoint's memory read path inject any content, or is
   it still "gate open, nothing to inject"
   (`models/mamba2_2_7b_memory/model.py` ~91-96)?

Checkpoint under test: `models/mamba2_780m_memory_mix/checkpoints/archive-bx1-mix16/step-333`
(BX1 endpoint, ~12.2M tokens — see `../experiments/EXPERIMENT_NOTES-20260724-014851.md` 12:05 UTC).

**Confidence:** everything here is `[M]` — measured locally, commands inline.

## 1. SSM interference capacity of the plain backbone

### Method

`sft/probe_recall.py --n-facts` sweep, `--ablation none` (memory→backbone
pathway disabled entirely: the **ablated column is the plain backbone**;
intact is the same checkpoint with the memory live, so mem-delta = intact −
backbone comes free). 4 probes/config × 3 seeds (1234/999/555), aggregated by
mean. Per config, one process (see §3 — a fresh-M numeric blowup otherwise
poisons later configs in the same process):

```
MODEL_NAME=mamba2_780m_memory_mix uv run --no-sync python -u probe_recall.py \
  --checkpoint ../models/mamba2_780m_memory_mix/checkpoints/archive-bx1-mix16/step-333 \
  --ablation none --gaps <48|512> --n-facts <nf> --n-probes 4 --seed <s>
```

(env vars as in `make probe-recall`, plus `HSA_OVERRIDE_GFX_VERSION=11.0.0`;
`--n-probes 8` and `16` both OOM on this 8 GB card — the per-slot fast-weight
buffers are 2.7B-scale.)

Each fact is exactly **12 tokens** (`[ROLE] The code for <label> is 4 8 2 1 3.`,
measured with the run tokenizer), so 512 tokens ≈ 42 facts. Decoded probe row
(seed 1234, n-facts 32 — labels are common single-token words, i.e. highly
confusable interference; note the fact-block → filler join and the query):

```
[USER] The code for that is 7 1 0 1 9.[ASSISTANT] The code for with is 0 1 1 5 3.[USER] The code for from is 0 0 0 5 9.[ASSISTANT] The code for this is 7 9 7 2 1. …
… 8 7 5 2.[USER] The weather in the valley stayed mild for most of the season.[ASSISTANT] A good soup starts with onions cooked slowly until they turn golden. …
[USER] What was the code for which?[ASSISTANT] The code for which is → target " 5 4 3 3 7"
```

### Results

Mean log-prob per code token (higher = better; uniform-random digits =
ln 0.1 = **−2.30**). "exact-code p" = e^(5·logprob), the chance of emitting
the whole 5-digit code.

| gap | n-facts | fact-block tokens | backbone | exact-code p | intact | mem-delta |
|----:|----:|----:|----:|----:|----:|----:|
| 48 | 1 | 12 | −0.089 | 64% | −0.011 | **+0.078** |
| 48 | 2 | 24 | −0.023 | 89% | −0.007 | +0.015 |
| 48 | 4 | 48 | −0.217 | 34% | −0.185 | +0.032 |
| 48 | 8 | 96 | −0.362 | 16% | −0.433 | −0.071 |
| 48 | 16 | 192 | −0.678 | 3.4% | −0.740 | −0.063 |
| 48 | 24 | 288 | −0.619 | 4.5% | −0.629 | −0.010 |
| 48 | 32 | 384 | −0.720 | 2.7% | −0.726 | −0.006 |
| 48 | 48 | 576 | −1.394 | 0.09% | −1.596 | −0.202 |
| 48 | 64 | 768 | −1.524 | 0.05% | −1.587 | −0.063 |
| 512 | 4 | 48 | −0.305 | 22% | −0.310 | −0.005 |
| 512 | 16 | 192 | −1.339 | 0.12% | −1.329 | +0.010 |
| 512 | 32 | 384 | −1.069 | 0.5% | −1.107 | −0.038 |
| 512 | 64 | 768 | −1.934 | 0.006% | −1.924 | +0.009 |

Raw per-seed lines: `sft/logs/probe-nfacts-sweep-bx1-step333-20260724.log`
(full per-run logs: `sft/logs/probe-2026072[45]-*.log`, 35 runs).

### Interpretation

- **The keystone number: backbone exact-code recall is effectively dead by
  16 competing facts (192 tokens of interference) at gap 48** — 3.4% for the
  full code, down from 64% at 1 fact — and is at chance-adjacent levels
  (≤0.1%) by 48 facts (576 tokens). Degradation starts at 4–8 facts.
- **SSM-defeating interference fits comfortably inside 512 tokens.** 16–32
  facts occupy 192–384 tokens. It doesn't even need 512; ~200 tokens of
  dense labeled facts is already past the backbone's capacity.
- Distance compounds it: at gap 512 the same fact loads score ~0.3–0.7 nats
  worse (16 facts: −1.34 vs −0.68). Interference + distance together defeat
  the backbone even faster.
- The degradation is smooth, not a cliff (there's a mild non-monotonic
  plateau at 24–32 facts, within seed noise — seeds spread ±0.2–0.3 at high
  n-facts with only 12 probes/config, so treat single-cell values as coarse).
- **The memory never picks up what the backbone drops:** mem-delta is +0.08
  at 1 fact (the regime where the backbone barely needs help), ~0 by 2–4
  facts, and ~0-or-negative everywhere the backbone is failing. Consistent
  with §2: there is nothing content-bearing to read.

## 2. Does the mix read path carry content?

### Method

`make read-diagnostic` on the same checkpoint (per-token ‖o_t‖/surprise +
residual capture), plus a ~100-line scratchpad probe that additionally
records the mix gate β and the actually-injected term ‖β·W_o(down(o_t))‖
per token (read_diagnostic doesn't capture either; scoped to this question,
not committed — script and output preserved as
`sft/logs/mix_gate_probe-20260724.py.txt` / `sft/logs/mix-gate-bx1-step333-20260724.log`):

```
MODEL_NAME=mamba2_780m_memory_mix make read-diagnostic ARGS="--ckpt \
  ../models/mamba2_780m_memory_mix/checkpoints/archive-bx1-mix16/step-333 \
  --probe-layer 10 --examples 3 --tokens 3072 --chunk-len 48 --save <captures.pt>"
```

(probe-layer 10 keeps the floor below the mix arm's layer-16 read/inject
point. Log: `sft/logs/read-diag-20260724-*.log`.)

### Results — 9,216 tokens of `data/train_memory_longalign.pt`

```
||o_t||   p0 10.16  p5 15.55  p50 21.53  p95 27.15  p100 44.59   mean 21.4  CV 0.167
surprise  p0 0.008  p50 0.139  p95 0.414  p100 1.163              mean 0.169 CV 0.787
corr(||o_t||, surprise) = −0.322
||residual entering layer 16||  p5 72.5  p50 128.5  p95 250.2     mean 141.3
```

Mix-gate probe (4,096 tokens, 2 examples, same dataset — β and the
actually-injected term, which read_diagnostic doesn't capture):

```
beta            p0 0.0001  p5 0.004  p50 0.051  p75 0.144  p95 0.417  p100 0.957   mean 0.110
||mix term||    p0 0.003   p5 0.091  p50 0.916  p75 2.064  p95 4.699  p100 12.07   mean 1.479
||mix|| / ||res@16||   median 0.62%   mean 0.99%   p95 3.2%
||o_t|| / ||res@16||   median 13.7%
```

Top-‖o_t‖ tokens decoded (the read peaks on newline/section boundaries in a
SAS deployment doc's table of contents, not on semantic content):

```
[2899] o_norm 31.04: '4 (TS1M3) and earlier . 59  >>\n<< Deploy the'
[9028] o_norm 30.69: '… any imputed savings from the partnership >>.<<  These arrangements are'
[1995] o_norm 30.43: 'Updating Product Documentation …… 27  >>\n<< Usage ……'
```

Also from the read_diagnostic run: layer-10 → layer-16 residual ridge
recoverability R² = 0.814 (the script's final q/k/v-cosine section crashes on
the 780m mix package — `models/mamba2_780m_memory_mix/model.py` doesn't
re-export `_rms_normalize`; harmless for this question, noted for whoever
next runs it).

### Write→read distance ≤ 48 tokens?

Not directly measurable with existing instruments — correlating *retrieved
content* with the write→read distance of specific facts would need per-fact
tracking that neither `probe_recall.py` nor `read_diagnostic.py` supports, and
building it was out of scope. Two proxies, both consistent with "nothing is
retrieved at any distance":

- §1's mem-delta is ~0 at gap 48 **and** gap 512 (facts ~60–800 tokens
  upstream of the query).
- The step-333 gist probes of record (`sft/logs/probe-bx1-step-333-none.log`):
  long-range (sleep-intact − sleep-recent) = +0.0001 (SEM 0.0011) — whatever
  minuscule gist-delta exists (+0.0037) is fully explained by the last ~576
  tokens.

### Interpretation

- **Not the 2.7B failure mode verbatim.** The prior observation was "gate
  open, o_t tiny". Here o_t is *not* tiny — ‖o_t‖ ≈ 21, ~14% of the residual
  norm entering layer 16 — and β is not pinned open: it sits mostly low
  (median 0.05) with real per-token variation up to ~0.96.
- **But what actually lands on the stream is ~1%.** The injected term
  β·W_o(down(o_t)) has median norm 0.9 against a residual norm of ~156 —
  0.6% (median) to 1% (mean) of the stream, p95 3.2%. The zero-init `W_o`
  woke up only far enough to whisper.
- **And the whisper carries no measurable content.** Everywhere §1's backbone
  is failing (8–64 facts), mem-delta ≈ 0 or negative; the only positive
  memory contribution (+0.08) is at 1 fact, where the backbone doesn't need
  it. Across sleeps, the probes of record show long-range gist of +0.0001
  (SEM 0.0011). o_t's variation is small (CV 0.17), *anti*-correlated with
  surprise (−0.32), and its peaks land on newline/section-boundary tokens,
  i.e. layout, not semantics.
- Verdict: the read path is numerically alive but functionally inert —
  "gate ajar, injection a content-free ~1% perturbation". The
  within-48-tokens hypothesis stays unresolved as a mechanism, but both
  proxies say no retrievable content at *any* distance on this checkpoint.

## 3. Incidental finding: fresh-M write dynamics go non-finite at inference

`probe_recall`'s floor pass (query from a fresh state, i.e. a fresh random M
under the trained write knobs, injection live) frequently goes NaN within a
few 8-token windows — `[nonfinite-write] … k_norm nan v_norm nan` — and in
roughly 1-in-3 configs the *prefix* pass from a fresh state blows up the same
way, NaN-ing intact/ablated too. M's init is drawn from the global RNG per
`init_state`, so incidence is stochastic per process. Consequences:

- Any multi-config `probe_recall` invocation on this checkpoint is unusable
  past the first config (the blowup's `[nonfinite-write]` spew is the tell);
  this session ran **one config per process** and retried NaN'd configs on a
  shifted seed (~5 retries in 35 runs).
- The floor column is unreliable on this checkpoint (NaN in ~half the runs,
  and finite values scatter −2.6 to −28 vs the theoretical −2.30) — the
  degradation curve in §1 stands on backbone-vs-uniform, not on the probe's
  floor.
- Worth knowing for training too: the same trained-knob/fresh-M combination
  is what every training sequence starts from; if the next run changes eta/
  theta/alpha dynamics, watch `GRAD_NORM`/`[nonfinite-write]` early.

## 4. Caveats

- 8 GB card: `--n-probes 4` × 3 seeds (12 probes/config) instead of one
  16-probe batch; high-n-facts cells are coarse (seed spread ±0.2–0.3 nats).
- Backbone = the BX1 checkpoint's backbone (base + its LoRA) with the memory
  pathway disabled — the relevant backbone for the next run's data design,
  but not the never-memory-trained control (`mamba2_780m`'s own checkpoints
  would answer that variant).
- Probe facts use `single_token_labels` skip=0 (held-out from
  `prepare_interference` training labels by construction).

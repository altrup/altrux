# Research note — 2026-07-22: memory-consolidation landscape & the "is M needed" reframing

Not a run log, not standing direction. A literature scan + reframing prompted
by a debrief question: *if the end goal is "generate training data during
sleep with M, then consolidate into weights," is the online memory M even
needed — and isn't the hippocampus a permanent episodic store, not just a
buffer?* Both turned out to be live questions in the current (2025–26)
literature, not open problems we'd be inventing.

**Confidence markers below:** [V] = verified against the paper's own
methods/results section (fetched, PDFs saved under this session's
tool-results/); [S] = from search snippet only, NOT yet read — treat as a
pointer, verify before relying. Next session: read the [S] items and the
saved PDFs before building anything on these claims.

## The reframing (the actionable idea)

We have been optimizing M for **direct read-out at query time** (gist-delta:
does injecting M's read improve continuation prediction across a sleep). We
are stuck at ~15% recovery (gist-delta ~+0.034 vs wipe cost ~0.19–0.22) and
two data recipes couldn't move the ceiling — see
`../discussion/DISCUSSION-20260722-stage2-readout.md`.

The CLS / consolidation framing says M's real job may not be direct read-out
at all, but to be a **good replay source for sleep-time consolidation into
weights**. Under that framing:

- The success metric changes from "M reads out well" (hard; where we're
  stuck) to "replaying from M consolidates as well as replaying the raw
  transcript" (easier; the 128-dim read-out straw stops being the
  bottleneck).
- The peak-height ceiling we can't beat may be a ceiling on the *wrong
  objective*.
- This does NOT kill M — under Complementary Learning Systems you need both
  a fast store and slow consolidation permanently, and the fast store's
  function is precisely to generate interference-free replay. It potentially
  *rescues* M from a ceiling that only exists for direct read-out.

Uncomfortable but honest: we have never run the baseline that would tell us
whether M beats trivial alternatives. Specced below.

## Neuroscience: hippocampus is contested, and the contest maps to our design

- **Standard systems consolidation** (Squire; Alvarez–McClelland): HC is the
  fast *temporary* store; memories transfer to neocortex via sleep replay and
  become HC-independent. → "M is a buffer, consolidate into weights."
- **Multiple Trace / Trace Transformation Theory** (Nadel & Moscovitch):
  detailed *episodic* memory stays HC-dependent ~indefinitely; only
  semanticized **gist** migrates to cortex. → HC is a permanent episodic
  store (the debrief's pushback — correct under this camp).
- **Complementary Learning Systems** (McClelland, McNaughton, O'Reilly 1995;
  updated 2016): HC = fast, sparse, pattern-separated, interference-resistant;
  neocortex = slow, interleaved, structural. Both permanent; HC's ongoing job
  includes generating replay so cortex learns without catastrophic forgetting.
  This is the framing that maps cleanly to ML and reconciles the two above:
  the gist half migrates (standard), the episodic detail persists in the fast
  store (MTT).

Implication: "M = disposable buffer" is only one of the two camps. If M's
durable role is episodic-detail store + replay source, then multi-sleep
durability (north-star ability #3, where 435 halves through one sleep) stops
being a thing M must solve alone — consolidation carries the load.

## Current literature (2025–26)

### Sleep-time consolidation for LLMs — our plan, already being published

- **"Language Models Need Sleep: Learning to Self-Modify and Consolidate
  Memories"** (arXiv 2606.03979) [V]. Built on **Hope / Nested Learning** —
  Behrouz et al.'s successor to Titans, i.e. THIS REPO'S ARCHITECTURAL
  LINEAGE. Multi-frequency memory blocks; sleep = transfer from fast
  (high-frequency) to slow (low-frequency) blocks via: (1) **parameter
  expansion** — add a low-rank expert to the slower MLP block before
  consolidating, to avoid interfering with existing knowledge; (2)
  **knowledge seeding** — on-policy distillation + RL imitation of the fast
  model into the expanded params (NREM-like); plus (3) **"dreaming"** —
  RL-curriculum synthetic data, gradient-scored for high impact, SFT on it
  (REM-like). Consolidation stages "Hope-1/2/3" show **monotonic improvement
  with more stages** (= the "longer sleep helps" knob, measured). Headline:
  near-perfect BABILong at **10M tokens** where ICL baselines degrade past
  256K. Baselines incl. ICL, EWC, InCA, Cartridges, SEAL, SFT. Stated
  limits: catastrophic forgetting from iterative self-improvement, SFT cost
  of dreaming, dependence on the Nested Learning foundation.
- **"Sleep-Like Memory Consolidation in LLMs" / "Language Models Need Sleep"**
  (arXiv 2605.26099) [S]. Different mechanism per snippet: at the eviction
  boundary, offline recurrent passes over accumulated context update **SSM
  fast weights via a learned local rule**, then clear the KV cache; deeper
  recurrence ("longer sleep") improves post-sleep reasoning. Closer to our
  SSM-state substrate than the Hope paper — read next.
- **"SCM: Sleep-Consolidated Memory for LLMs"** (emergentmind 2604.20943) [S].

### Surprise-driven replay — a better home for our surprise signal

- **"SuRe: Surprise-Driven Prioritised Replay for Continual LLM Learning"**
  (arXiv 2511.22367, Nov 2025) [V]. Surprise = prediction error; used to
  **prioritize which buffered real examples get replayed** during
  consolidation (not synthetic; not a read-gate). Neuro/RL grounding:
  unexpected events drive consolidation. Eval on task sequences
  (GLUE/SuperGLUE) with retention + forward/backward transfer; specific
  numbers not extracted [partial]. Takeaway for us: surprise's defensible
  role is "**what to consolidate**," not "when to inject" — supports the
  debrief's skepticism that surprise belongs in a read-gate.

### Hippocampus-as-persistent-store architectures

- **HEMA** (arXiv 2504.16754) [S] and **ZenBrain** (arXiv 2604.23878) [S]
  build the fast/episodic store as a *persistent* layer, matching the MTT
  reading rather than "temporary buffer."

### Evaluation — directly relevant to our "is gist-delta the right ruler" pain

- **"Beyond Perplexity: A Behavioral Evaluation Framework for
  Deployment-Memory Claims in LLM Test-Time Training"** (arXiv 2607.00368)
  [V]. Argues **perplexity / continuation-loss are incomplete proxies** for
  memory — they measure token prediction, not whether specific info was
  stored and retrievable across a context boundary. Proposes behavioral
  probes: exact factual recall of earlier-context items, consistency across
  context resets, and separating true memory from in-context/attention
  retrieval. **This is an argument that our headline metric (gist-delta, a
  continuation-loss delta) is exactly the incomplete kind**, and dovetails
  with our own north-star ability #2 (specific-fact recall on demand = ~0 at
  435). Candidate source for a better ruler than ultrachat-2048.

### Lineage / surveys

- Titans (arXiv 2501.00663) — this repo's basis. Continual-learning survey
  covering Titans + End-to-End TTT: shuaichenchang.github.io/posts/2026/
  continual-learning-1/. End-to-End TTT (Dec 2025), In-Place TTT
  (2604.06169), PERK (ICLR 2026) — TTT descendants [S].

## Implications for our program (proposed, not yet in standing direction)

1. **Lit-review is the highest-leverage next local task** — ahead of
   speccing our own consolidation baseline. The sleep-consolidation papers
   (esp. 2605.26099's SSM-fast-weight mechanism and the Hope paper) have run
   versions of the baseline we were about to build. Read before reinventing.
2. **Consolidation baseline, three arms** (specced last, on the box or a
   strong local slice): on the same gist-probe setup (prefix → sleep →
   continuation), compare (a) **M-direct-read** (current), (b)
   **consolidate-from-transcript** — brief LoRA/adapter update on the prefix
   transcript during "sleep," score continuation with state wiped, (c)
   **consolidate-from-M-replay** — replay generated from M, same update. If
   (b) ≫ (a), M's value is latency/cost only; if (c) ≈ (b) ≫ (a), M earns its
   place as a replay source. Prediction: (b) beats (a), possibly by a lot.
3. **Surprise → consolidation prioritizer** (SuRe), not a read-gate. Fold
   into any hybrid design.
4. **"Longer sleep helps"** (Hope-1/2/3 monotonic) is a free empirical lever
   if we go hybrid.
5. **Better ruler**: evaluate the Beyond-Perplexity probes as a replacement
   for the ultrachat-2048 gist ruler (the carried open question in the
   stage-2 DISCUSSION).

These do NOT override tonight's committed direction (B1 plateau, B2a
window-4) — those answer "is the online mechanism improvable," which a hybrid
needs regardless. This note sits ABOVE that as the "what are we optimizing M
for" question, to be taken up at the next debrief.

## Integration point: we may be using the weakest variant [V from Titans]

How the memory READ is wired into the backbone, from Titans (2501.00663):

- **MAC (Memory as Context):** retrieved memory concatenated as tokens
  before the first layer → flows through the WHOLE network, attention
  chooses relevance. Strongest on long dependencies.
- **MAG (Memory as Gate):** memory is a parallel branch, blended with a
  precise-recent branch via a learned gate. Strong; clean short-vs-long
  separation.
- **MAL (Memory as Layer):** memory as a stacked layer, contribution baked
  in before attention. Titans found it WEAKEST.

**Our design = gated-delta merge into the mamba SSM state at layers ≥22 —
closest in spirit to MAL, the weakest variant.** We get neither MAC's
full-depth "attention chooses" nor MAG's clean recent-vs-remembered gate.
The read enters deep, so only the layers above 22 compute over it — least
room to reason over what was retrieved, i.e. we're on the wrong side of the
compute-depth axis (below). NOT directly measured (MAC>MAL is transformer,
not mamba), but suggestive.

Actionable lever: **inject the read LOWER** (nearer the embedding, so it
propagates through more SSM layers) — the MAC-analog for a mamba stack.
Separate knob from where q/k/v are GENERATED (the layer-21 front-end move).
Corollary: since Mamba2's own SSM state is ALREADY a linear associative
test-time memory (via State Space Duality), injecting M into that same state
makes two associative memories share one substrate — structurally the
awake-mem / gist-vs-verbatim competition we observe. A MAG-style *separate
branch + gate* matches the short/long split better than fusing them.
(Frozen backbone is NOT essential — an experiment, not load-bearing; the
Titans-reimpl frozen-insufficiency finding is de-weighted accordingly.)

## Compute over the read, not just bandwidth [V from 2605.26099]

2605.26099's central finding, independently echoing our own north-star note
("capacity is NOT the bottleneck"): **computation, not storage, is the
bottleneck.** Vanilla hybrids fail on deep reasoning DESPITE sufficient
fast-weight capacity — performance drops with reasoning DEPTH at fixed
length. Storage holds the fact; the model just can't chain operations over
it. Sleep (N recurrent offline passes) helps precisely by adding compute
depth over consolidated content (Hope N=1→4 monotonic; e.g. GSM-Infinite
6-op 0.742→0.812).

Reframes our stage-2 "read-out bandwidth" suspect: the ceiling may be
compute-DEPTH over the read (one shallow gated lookup per window), not
bandwidth-WIDTH (128-dim straw). Different fix — more passes / inject-lower
/ longer sleep, vs. widening. Add "compute-depth-over-the-read" as a
first-class ceiling suspect, arguably ahead of widening.

## Design sketch: M→weights consolidation via dream + un-write

A teammate proposal (worth recording; NOT for the next run). Move content
from M into the backbone WEIGHTS (hippocampus→cortex systems consolidation),
freeing M:

1. **Dream:** generate a sequence with M in place (SSM randomized so the
   dream can't echo short-term — forces M to be the source), logging every
   q→v M injects.
2. **Un-write:** reverse-gradient-descent those logged q→v OUT of M (push
   M(q) off v) → M emptied of that content.
3. **Consolidate:** train the trainable weights on the dream with SSM
   cleared and M emptied → content can't come from M or SSM, so it must
   land in the weights.

Net: memory MOVED (not copied) M→weights; M freed. Elegant points: the
un-write is what forces the transfer (else the model keeps reading from M);
and offloading the durable load to weights (full depth+bandwidth) routes
AROUND M's read-out ceiling — M's job shrinks to "generate a faithful dream,"
not "be a queryable long-term store" (the reframing this whole note argues
for).

**Two potentially-fatal risks the sketch must address:**

1. **Catastrophic forgetting.** Step 3 trains shared weights on dreams with
   no general-data rehearsal → drifts toward "good at dreams, worse at all
   else." Every consolidation paper mixes replay. Mitigation: interleave
   general corpus, or consolidate into an isolated per-user LoRA (reintro-
   duces storage/routing cost).
2. **Self-distillation collapse.** The dream is generated from LOSSY M (gist
   ≫ verbatim at 435) but trained on AS ground truth; looped, errors compound
   (photocopy-of-photocopy → model collapse). The un-write makes it worse:
   once M is emptied, the reference is gone.

Also: un-write may not be surgical (distributed store → damages neighbors;
measure by un-writing a set then probing unrelated recalls).

**Cheap precondition, testable now on 447-T3 (does NOT need the box):** can
M even generate a faithful dream? Prediction from our gist-vs-verbatim data:
gist-faithful, fact-lossy. If so, the loop bakes wrong specifics into
weights and needs the transcript in the loop anyway — collapsing toward the
transcript-replay baseline. See the "M readout/dream-fidelity probe" local
item in the stage-2 DISCUSSION. The permanent control for the whole idea is
**move-through-M vs move-through-transcript** (keep raw data, distill
directly) — M earns its place only if compression beats the fidelity it
costs. Where it fits: phase-3 recombination (deployment-time consolidation),
not a way to train M better now.

## Caveats on this note

Single search pass + fast fetch of a handful of papers (2606.03979,
2511.22367, 2607.00368, 2510.09551, 2605.26099) via a summarizer model —
not a careful read. [S] items are snippet-level. Integration-variant and
compute-not-storage claims are [V] at summary level (MAC>MAL is transformer
evidence, an analogy for our mamba state-injection, not a measurement).
PDFs saved locally this session. Verify before any of the above becomes
standing direction.

# Experiment notes — 2026-10-28 02:04 UTC: LAMA-CKL sixth box (dreamer session, arm a)

Governing note: `notes/discussion/DISCUSSION-20261024-lama-ckl-fifth-box-debrief.md`
(§3.5 item 3: the dreamer session; §3.2 arm (a) mechanism block; §3.3 corpus).
Box branch `box/20261028-015757`, GH200 480GB (97871 MiB, torch 2.11.0+cu128).
Model `mamba2_2_7b`. Session plan: corpus regeneration (box, §4.56) → box alignment
gate on `ultra20k-v4` at 512 tokens, fp32 forward (the verdict on arm (a)) →
arm (a) dreamer, two rounds → final-checkpoint instrument/prod pass → shutdown.
No cell runs this session (the cell session is §3.5 item 4, after the dreamer).

## Preflight

- Box arrived with: `models/mamba2_2_7b/checkpoints/ultra20k-v4/epoch-1/step-9594`
  (sha256 of `trainable.pt` `7ba7972b308c70c777d4f592e46a781b4b9dad22717cec247a6f10821a054ad9`
  = `sft/experiments/lama_ckl/warmstart.sha256`), the v4 split
  `.cache/lama_ckl/mamba2_2_7b_ultra20k-v4/`, `upstream/finetune_qlora.pkl`,
  `sft/data/dolly_context.jsonl`. `runs/` empty (v4 baselines archived at home, §6.8).
  No `.cache/LAMA`, no `.cache/huggingface`, no `dreamer_*` corpus files (expected:
  the corpus is regenerated on the box, skill table row "Regenerate the dreamer corpus").
- 02:05 `final` link recreated: `ln -s epoch-1/step-9594 models/mamba2_2_7b/checkpoints/ultra20k-v4/final`.
- Working tree: `sft/uv.lock` modified by setup's re-resolve; left uncommitted.
- 02:06 `work` window `lama`: LAMA archive download + unzip to `.cache/LAMA/data`
  (log `sft/logs/lama-download-20261028.log`).
- 02:08 `work` window `hf`: prefetch `rajpurkar/squad_v2` train, `HuggingFaceH4/ultrachat_200k`
  train_sft, `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` into `.cache/huggingface`
  (log `sft/logs/hf-prefetch-20261028.log`).

## Command log (verbatim)

- 02:07 LAMA archive landed (`.cache/LAMA/data`: ConceptNet, Google_RE, Squad, TREx, relations.jsonl; 367 MB). HF prefetch EXIT=0 (4.9 GB).
- 02:07 preflight tests, `work` window `tests`, log `sft/logs/preflight-tests-20261028.log`:
  `make test ARGS="tests/experiments/lama_ckl/test_dreamer.py tests/experiments/lama_ckl/test_alignment_gate.py tests/experiments/lama_ckl/test_dreamer_corpus.py -q"`
  (the target prepends `tests ../models/tests`, so this is the whole suite plus those three).
- 02:08 **`data/dreamer_fixed_sets.json` did not arrive with the upload**, and the generator
  draws fresh gate rows when that file is absent (`dreamer_corpus.py:662`), which would
  orphan the analyst's 45 hand-written dreams in `diagnostics/gate_dreams.json` (keyed by
  passage id = sha256(passage text)[:12]). Decision: rebuilt a minimal fixed-sets file
  holding only the five `gate` rows (passage_id, title, first answerable question with
  squad_id/question/answer) from the SQuAD 2.0 train split in the HF cache, with
  `/tmp/.../scratchpad/rebuild_gate_rows.py` (throwaway; the five pids are gate_dreams.json's keys).
  The generator then applies its own keep rule (each row still unknown to v4 by
  generate-and-match AND the question is the passage's first answerable one); a miss
  prints `WARNING: fresh gate draw` and is stop-and-report (no new dreams can be written here).
  The five rows: 4b5826f9e571 PlayStation_3 ("quietness"), 296e9cab4705 Wood ("supporting
  material"), f90b15da807b BeiDou ("December 2011"), e62f3cd7b9ae Protestantism ("around 1790"),
  bb6c941eca4d Order_of_the_British_Empire ("1940"). Every other fixed set (prod, held-out,
  abstention, warm-start held-out) is drawn fresh on the box at `--seed 0`, as the skill
  table's "regenerate the corpus on the box" step intends.
- 02:08 corpus generation, `train`:
  `cd ~/altrux/sft && make lama-ckl-corpus ARGS="generate --lama-root ../.cache/LAMA/data"`
  Killed at 02:15: the generate-and-match scan ran at 2.1 passages/s (ETA 99 min, 9 GB,
  12 % GPU utilisation: latency-bound at the default `--batch-size 8`). Relaunched:
  `cd ~/altrux/sft && make lama-ckl-corpus ARGS="generate --lama-root ../.cache/LAMA/data --batch-size 64"`
  (batch size changes compute only; the draws are seeded.)
  At 64: 6.4/s, ETA 32 min, 30 GB, still 12 % utilisation. Killed at 02:18 and relaunched at 128:
  `cd ~/altrux/sft && make lama-ckl-corpus ARGS="generate --lama-root ../.cache/LAMA/data --batch-size 128"`
- 02:13 preflight tests EXIT=0: 840 passed, 1 xfailed (307 s).
  At 128: 7.9/s, ETA 26 min, 54 GB, 12 % utilisation (the 48-token greedy loop is
  launch-latency-bound; batching buys little past 64). Left to run.
- 02:49 corpus generation EXIT=0 (log `sft/logs/lama-ckl-corpus-20261028-021804.log`, 31 min, of which
  27 min the generate-and-match scan). Outputs `sft/data/dreamer_corpus.jsonl` (200 MB),
  `dreamer_fixed_sets.json` (2.5 MB; prod 20, gate 5, held_out_facts 50, held_out_warm_start 200,
  `warmstart_sha256` = the pin), `dreamer_abstentions.json`.
- 02:50 box alignment gate on v4 (§3.5 item 2), `train`:
  `cd ~/altrux/sft && make lama-ckl-gate ARGS="--init-adapter ../models/mamba2_2_7b/checkpoints/ultra20k-v4/final --tokens 512 --fp32-forward --output ../.cache/lama_ckl/gate/ultra20k-v4"`
  Retained-pool pass first: 2,000 questions at 1.7/s (fp32 forward, batch 4), ETA 19 min, 12.7 GB.

## Corpus readings (from the generator log; the §3.3 "sanity-check the artifact" read)

- LAMA (subject, object) pairs 13,420. SQuAD: 19,029 passages, 12,862 LAMA-clean (67.6 %),
  12,479 with ≥ 2 answerable questions; 48,837 questions dropped by the pair rule (top first
  matches: science/science 2,472, australia/australia 2,130, france/french 1,117).
  Conversation candidates: 1,000 kept of 1,567 read (560 LAMA-excluded, 41 acknowledgement
  boundaries, 31 capped, 7 without a valid boundary); 2,020 valid boundaries. Warm-start
  UltraChat 12,997/20,000 LAMA-clean (65.0 %); packer would have dropped 10 of 20,000 whole, 5 of
  the 12,797 pool rows, 0 of the 200 held-out rows. Dolly plain 984/1,771 clean (55.6 %).
- Invariants: `LAMA overlap over the written rows: 0`, `passages shared across kinds: 0`,
  `scorer ids != format_conversation ids: 0`, `per-token correctness != scorer accuracy: 0`. All as required.
- v4 first-question scan, generate-and-match: 12,479 scored, **1,194 known (9.6 %)**; teacher-forced
  accuracy histogram 0.0: 4,359 … 0.5: 2,334 … 1.0: 87. Agreement (any-token, first-token, generated):
  (T,F,F) 7,081, (F,F,F) 3,899, (T,F,T) 627, (F,F,T) 460, (T,T,F) 305, (T,T,T) 107 — the any-token
  rule calls 8,120 "known" where generation confirms 734; the generated rule is the one in force (§4.56).
- **Gate rows kept**: all five previous gate rows unknown to v4 by generate-and-match and each is its
  passage's first question (replies quoted in the log: "…quality of the PS3 Super Slim's screen",
  "…flooring, walls, and roofing", "…began operating in China in 2011", "…early 19th century",
  "…from the year of the Order's creation"). `gate_dreams.json` stands.
- Fact pool 12,363 passages / 57,047 answerable questions; 37.2 % of questions have no capitalised
  word after the first (the "unbound question" reading). Held-out v4-best-known teacher-forced
  accuracy range 0.000..1.000 (known by generation, not necessarily by forced tokens).
- **Abstention set: 709 phrasings from 8,520 candidate replies (labels: answers 6,629, cannot 1,181,
  unknown 710).** Composition of the 709 by a crude read (own count over `dreamer_abstentions.json`
  `phrases`): **331 end in "?"** (the model echoing an unrelated question back, e.g. "In what month did
  Tucson get the most rain?"), 189 carry a negation/unknown word ("I don't have the information.",
  "The population of the Marshall Islands in 1862 is not known."), **212 neither** (confident
  statements and bare answers: "In 1824", "India", "Apple, Samsung Electronics and Sandisk"). The
  five decoded abstention rows in the log are all non-abstentions (a date "On May 1, 2020", three
  question echoes, one statement about Jehovah's Witnesses). So the classifier's "does not know"
  label admits mostly non-abstentions; about a quarter of the set reads as an abstention.
  Abstention rows written: 203 from 42 passages, 174 distinct replies.
  **Decision: continue.** The classifier rule is registered (§3.3, §4.56, decided 10-27, "no stem
  list anywhere"), the decoded samples were a debrief read before scheduling, and the experimenter
  has no authority to change a corpus rule. Consequences to read at the debrief: (1) the abstention
  g_old hinge protects the loss of `[USER] unanswerable-q [ASSISTANT] <random echo or wrong
  statement>`, i.e. mostly a noise direction, not the abstention behaviour; (2) the over-abstention
  detector (same classifier) will count question echoes and bare wrong answers as abstentions, so
  the row-4b rate is inflated and its round-to-round change is the only usable reading.
- Decoded samples (fact, conversation, prod, gate, held-out, dolly_plain, warm_start): shapes as
  specified — fact C is `[USER] q [ASSISTANT] a<eos><eoc>`, C⁻ the same q with another row's
  answer (e.g. "Methodism" vs "corruption"); conversation W ends on an assistant turn + EOC, C is
  the follow-up + the real reply, C⁻ the follow-up + another row's reply; no garbled splices seen.
  (Five fact samples drew C⁻ from among the same five — the pairing is a derangement within the
  printed set, fine.)
- Second read of the same samples by a haiku-class subagent (the data-sanity rule): agrees — every
  kind matches its shape except abstention, where none of the five replies abstains. Its other
  flags are source-text artefacts, not generator faults: the Cyrillic "с" in "сuprum"/"сyprium" and
  the dropped first letters "hat happens…", "alvia rosmarinus" are in the SQuAD/Dolly source rows;
  `Hairdresser: "…"` / `User: "…"` are UltraChat role-play content inside an assistant turn.
- 02:56 utilisation note: the gate's retained-pool pass decodes 2,000 greedy replies at batch 4
  (1.3/s, ~26 min, 12.7 GB, low utilisation). `alignment_gate.py --batch-size` is shared with the
  g_old backward, so it was not raised mid-verdict. A separate decode batch flag for the pool pass
  (as `dreamer_corpus.py --batch-size 128` did, 7.9/s) would save ~20 min per instrument run (there
  are three or more per dreamer track). Nothing in the §3.5 order can run beside the gate: the
  dreamer needs `gate.json`, the cells need α and c, and the floor step measures cross-process spread.

## Box gate on v4 (§3.5 item 2), live readings

- 03:16 retained-pool pass (vii), fp32 forward: 2,000 candidates → **known 166, unknown 1,834** by
  generate-and-match; teacher-forced accuracy histogram 0.0: 739, 0.5: 365, 1.0: 14; known rows by
  correct answer-token count {0: 66, 1: 40, 2: 27, 3: 15, …} (66 generated-known rows have no
  teacher-forced-correct answer token — the two rules disagree often, as the corpus scan did).
  Written as `data/retained_pool-7ba7972b-fp32.json`.
- 03:17 fit: 512 tokens, batch 1, forward+backward peak 12.53 GB (fp32). g_old reference passes
  next (67 GB resident at batch 4, 19 % util).
- 03:27 gate EXIT=0 after 37 min (log `sft/logs/alignment_gate-20261028-025024.log`; `gate.json` in
  `.cache/lama_ckl/gate/ultra20k-v4/`). **Verdict: FAIL, 0 of 5 rows pass** (the rule needs 4 of 5).
  Instrument values the script still reports: cell α 0.01 (the sweep top, "no edge found" on the two
  rows that reached (iv)), c 1.2546 (row 4b5826f9e571's ‖G‖), h 72.39, κ 0.724 (status "fit"),
  intercept 3.5e-4. θ statistics: 21,323,264 coordinates, RMS 0.00918, median |θ| 0.00626, median
  bf16 spacing 3.05e-5 (the §3.4 figure, confirmed), zeros 0. R drawn 34/33/33.

### Gate table (the script's summary, verbatim)

| row | alpha | edge | ‖G‖ | L(C) | floor | pred@rung | stat | rank/84 | s order | pass |
|---|---|---|---|---|---|---|---|---|---|---|
| 4b5826f9e571 | 0.01 | False | 1.255e+00 | 7.377 | 1.59e-04 | 1.22e-02 | -1.26e-03 | 60 | False | False |
| 296e9cab4705 | None | True | 5.013e-01 | 11.995 | nan | nan | nan | - | - | False |
| f90b15da807b | 0.01 | False | 2.996e-01 | 7.484 | 2.15e-04 | 2.86e-03 | -7.90e-03 | 78 | False | False |
| e62f3cd7b9ae | None | True | 3.440e-01 | 4.623 | nan | nan | nan | - | - | False |
| bb6c941eca4d | None | True | 3.446e-01 | 8.804 | nan | nan | nan | - | - | False |

### Reading of the fail (from the log and `gate.json`; drop > 0 = L(C) fell)

1. **Three rows fail at the sweep (i) because the revisit batch's step RAISES L(C).** On
   296e9cab4705 (Wood), e62f3cd7b9ae (Protestantism) and bb6c941eca4d (OBE) the first-order
   prediction lr·⟨G, ∇L(C)⟩ is negative at every rung and the measured drop tracks it closely
   (bb6c at 1e-2: measured −9.86e-3, predicted −9.82e-3; 296e: −8.97e-3 vs −9.17e-3; e62f: −2.29e-3
   vs −2.30e-3). The instrument is self-consistent (finite difference = first order to ~1 %); the
   result is the model's. `alpha_rule` requires `predicted > 0`, so these rows get α None and
   "edge" (no rung passes). G here is the mean over the revisit dream and the row's nine 512-token
   pool samples, so it is 90 % filler by count; the fillers are v4's own EOC dreams from the wake
   state (decoded in the log: UltraChat-shaped Q&A unrelated to the passage).
2. **The two rows that reach (iv) show a content-blind reward.** Per-dream scalars on
   4b5826f9e571 (PS3) and f90b15da807b (BeiDou), 18 hand-written dreams:
   - s_contrast is **positive for all 18** (every dream, including the three "unrelated"
     UltraChat-shaped ones, is predicted to lower the contrast loss).
   - The acknowledgement dreams score **higher** than the revisits: PS3 revisits s = 0.273, 0.068,
     0.297 vs acknowledgements 0.891, 0.138, 0.728 and unrelated 0.408, 0.171, 0.002; BeiDou
     revisits 0.091, 0.095, 0.159 vs acknowledgements 0.580, 1.021, **2.568** and unrelated 0.932,
     0.188, 0.130. "s ranks revisits first" is False on both rows.
   - The ranking follows the gradient norm, not the content: ‖∇ᵢ‖ revisit 0.93–2.33,
     acknowledgement 3.49–9.13, unrelated 1.27–6.44 (norms_by_kind means: PS3 1.83 / 6.54 / 3.59,
     BeiDou 1.11 / 6.51 / 3.84). Cosines are flat across kinds (cos_contrast revisit 0.10–0.31,
     acknowledgement 0.09–0.33, unrelated 0.00–0.40). Under uniform 1/N per-token weights a
     10–15-token acknowledgement has 30–50× the per-token weight of a 500-token revisit, and the
     `[USER]` sentence of each acknowledgement ("Spong reviewed the Super Slim PS3 and liked it.",
     "China's Beidou satellite system went operational after testing.") is a scored position that
     carries the passage's topic. This is the §3.2 "known danger" (a short dream carries the same
     total weight as a long revisit) and the §4.25/§4.40 case, measured.
   - Discrimination (iv), at rung 3e-3 with nine fillers shared: PS3 revisit batches dropped L(C) by
     0.0116, 0.0092, 0.0128 against acknowledgements 0.0172, 0.0100, 0.0154 and unrelated 0.0137,
     0.0099, 0.0086 → statistic −1.26e-3, rank 60/84, p 0.71; BeiDou revisits 0.0038, 0.0026,
     0.0039 against acknowledgements 0.0089, 0.0123, **0.0290** and unrelated 0.0124, 0.0042, 0.0010
     → statistic −7.9e-3, rank 78/84, p 0.93. The acknowledgement-minus-unrelated statistic is
     positive on both (+3.5e-3, +1.08e-2). Noise floors 1.6e-4 and 2.2e-4 (worker drops identical
     within a process, as §1.5 predicted; the cross-process spread is tiny under the fp32 forward)
     against predictions at the rung of 1.22e-2 and 2.86e-3, so the measurement is not the limit.
3. **The hinge quantities (oᵢ)**: o_facts is mixed in sign across dreams (−0.066 … +0.142),
   o_abstention is negative for 17 of 18 dreams (every dream's step raises the abstention-row loss a
   little; given the abstention set's composition above this is a noise direction), o_warm_start is
   small and positive for all 18.
4. **Wake replies** (sampled at 1.0, frame from the set): PS3 "Thanks for the details about the PS3
   Super Slim. What is the retail price?"; Wood "Also, to highlight that wooden furniture is made
   mostly of wood."; BeiDou "Got it, thanks for passing that along."; Protestantism "The Second Great
   Awakening gained momentum by 1800, reached its peak by the late 1840s, and it was past its peak by
   the late 1840s." (a revisit as the reply, without the answer); OBE "Okay I've taken that in."
   Two of five are acknowledgements, one asks a question, one restates, one is a non sequitur.
5. L(C) vs L(C⁻) per row: PS3 7.38 / 15.06, Wood **11.99 / 7.23**, BeiDou 7.48 / 10.57, Protestantism
   4.62 / 4.52, OBE **8.80 / 6.34** — on two rows v4 finds the random wrong answer more likely than
   the true one, and on a third they are equal. The three rows where the revisit step raises L(C)
   are the three where L(C) ≥ L(C⁻) − 0.1.

**Decision: STOP-AND-REPORT.** §3.5 item 2: "Fail → the reward is not live on this model; §3.2 (a)
does not proceed, and the fallback is §4.8's supervised bridge first"; "the box run is the first 30
minutes of the dreamer session and aborts it on a fail". No dreamer round, no cell, no split runs
this session. The fallback is a debrief decision, not the experimenter's. Shutdown checklist follows.

## Closing (shutdown checklist, stop-and-report)

**What ran** (02:04–03:35 UTC, ~1.5 h of GH200): LAMA archive and HF prefetch; preflight tests
(840 passed); the dreamer corpus regenerated on the box with the five analyst gate rows kept
(`dreamer_corpus.jsonl` 200 MB, `dreamer_fixed_sets.json`, `dreamer_abstentions.json`); the box
alignment gate on `ultra20k-v4` at 512 tokens under the fp32 forward, including the retained-pool
pass (`retained_pool-7ba7972b-fp32.json`, 166 known / 1,834 unknown). **The gate failed 0 of 5**, so
the dreamer (arm a), the final-checkpoint prod pass, and every cell-session step did not run. No code
was changed; no protected file was edited; nothing was committed but this note.

**What the diagnostics show** (detail above): (1) on three of five rows the registered cell step on a
revisit batch raises the true-reply loss, with the first-order prediction matching the measured
change to about 1 %, so the instrument is sound and the sign is the model's; (2) on the two rows
where the step helps, the reward does not separate content from shape: every hand-written dream has
a positive sᵢ, acknowledgements outscore revisits by 3–16× because their gradient norm is 3–6×
larger under uniform 1/N weights (their scored `[USER]` sentence names the topic), and the
permutation test ranks revisits 60th and 78th of 84; (3) the abstention set the classifier admitted
is about a quarter abstentions (331/709 question echoes, 212 confident statements), so the
abstention hinge and the over-abstention detector read a noise direction.

**What to discuss at the debrief:**
- §3.5 item 2's own fallback: "§4.8's supervised bridge first" (one epoch of SFT on recall and
  question rows). The gate measured exactly the dead/content-blind reward the §4.8 bridge exists for.
- Whether the sweep batch (one dream + nine v4 pool fillers) is the right instrument for α when the
  fillers set the sign: on three rows G·∇L(C) < 0 was a property of the fillers' step, which is the
  cell's step, so the reading stands for the cell — but it says the v4 cell at one document per wake
  would move away from the answer on three of five documents before any dreamer is involved.
- The uniform-1/N weighting lets a 10-token acknowledgement outweigh a 500-token revisit per
  gradient norm (§3.2 known danger, now with numbers); whether the gate's acknowledgement dreams
  should name the topic in their `[USER]` turn, since that turn is a scored position.
- The abstention labeler: the entailment classifier's "does not know" label is wrong on ~3/4 of the
  set; "no stem list anywhere" (§3.3) and this reading are in tension.
- L(C) ≥ L(C⁻) on three gate rows: v4 rates a random wrong answer as likely as the true one on
  these unknown facts, which is what "unknown" means, but it also means ∇L(C) − ∇L(C⁻) has no
  reason to be a learning direction for the step the fillers take.
- Instrument speed: the gate's retained-pool decode at batch 4 is ~26 min of its 37; a decode-batch
  flag (as `dreamer_corpus.py --batch-size 128`, 7.9/s) would make each instrument run ~15 min.
- `dreamer_fixed_sets.json` did not reach the box (the `LAMBDA_DATA_ARTIFACTS` line of §6.8 was not
  in effect at launch; only `dolly_context.jsonl` arrived); the five gate rows were rebuilt from
  `gate_dreams.json`'s keys and validated by the generator's own keep rule. The prod, held-out and
  abstention sets of this corpus are therefore a fresh box draw at seed 0, not the local ones the
  analyst and teammate read; the decoded samples are in `lama-ckl-corpus-20261028-021804.log`.

**Resume point for a next session:** nothing to resume. The box-built corpus, retained pool and
`gate.json` are home (receipt below) and can be re-uploaded; the v4 `final` symlink must be
recreated again on any new box.

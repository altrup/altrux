# Notes index

## Status (2026-09-25)

- **Active:** continual learning under the LAMA-CKL wake/dream protocol. This is the only governing direction.
- **Retired:** the neural-memory model (M, erase-on-read, state/token-mix integration). Its notes are historical record, not roadmap. See the known-bugs list before reviving any of it.
- **Side interest, not scheduled:** internal thinking by recurrent depth with KL-convergence exit (open idea in the [2026-07-23 discussion](discussion/DISCUSSION-20260723-780m-integration-screen.md)). Nothing built; no run planned.

## Current reading path

Read [the rented-GPU handoff](experiments/EXPERIMENT_NOTES-20260823-lama-ckl-gpu-handoff.md), [the LAMA-CKL wake/dream protocol](discussion/DISCUSSION-20260823-lama-ckl-wake-dream-protocol.md), then [the modern LLM CL evaluation review](research/RESEARCH-20260823-modern-llm-continual-learning-evaluations.md), [the benchmark selection review](research/RESEARCH-20260821-continual-knowledge-benchmark-selection.md), and [the dream-distillation prior-art review](research/RESEARCH-20260805-dream-distillation-prior-art.md). The [earlier six-wake design](discussion/DISCUSSION-20260811-multisleep-adaptive-wake-and-agent-migration.md) is retained as engineering evidence, not the next science target. The [2026-08-14 generalizability discussion](discussion/DISCUSSION-20260814-generalizable-cl-evaluation-design.md) records a rejected bespoke benchmark proposal and is not governing direction. Open older notes only when one of these documents links a decision or a live question needs its source.

## Standing lists

- [Known bugs, 2026-09-24](KNOWN_BUGS-20260924.md): Unfixed findings from the protected-files review, grouped by area; check before reviving the memory model, erasure, or chains pipeline.

## Discussion notes

- [2026-08-23 — LAMA-CKL wake/dream protocol](discussion/DISCUSSION-20260823-lama-ckl-wake-dream-protocol.md): Freezes the trusted benchmark, 30-cycle wake/sleep mapping, EOC-plus-instruction dream transition, secondary dream diagnostics, and implementation docket.
- [2026-07-20 — gist eval](discussion/DISCUSSION-20260720-gist-eval.md): Reframes the failed exact-code probe as a possible gist mismatch and specifies a continuous-text cross-sleep gist evaluation.
- [2026-07-21 — peak reproducibility](discussion/DISCUSSION-20260721-peak-reproducibility.md): Establishes step 435 as a reproducible episodic-gist checkpoint and orders tests for generalization, reproduction, and continuous improvement.
- [2026-07-22 — stage 2 readout](discussion/DISCUSSION-20260722-stage2-readout.md): Moves stage 2 from peak chasing to readout bandwidth, write fidelity, and compute-depth diagnostics around checkpoint 447-T3.
- [2026-07-23 — 780M integration screen](discussion/DISCUSSION-20260723-780m-integration-screen.md): Pivots to a faster 780M platform to compare state injection with token-mix integration points.
- [2026-07-24 — data structure and consolidation](discussion/DISCUSSION-20260724-data-structure-and-consolidation.md): Finds malformed or weak retention signals in the data and proposes solvability filters, repaired splices, distance curricula, and consolidation framing.
- [2026-07-24 — next-run plan](discussion/DISCUSSION-20260724-next-run-plan.md): Locks the 780M model, four-part data mixture, loss weights, and trainer/data fixes for the next run.
- [2026-07-25 — CL sleep analysis](discussion/DISCUSSION-20260725-cl-sleep-analysis-and-filter-testc.md): Defines signal-gated memory decay, the transcript null, behavioral multi-sleep evaluation, and filter test C.
- [2026-07-25 — implementation state](discussion/DISCUSSION-20260725-implementation-state-and-box-handoff.md): Records the shipped trainer, generators, filters, pilot calibration, regenerated artifacts, and ordered box handoff.
- [2026-07-30 — SSM consume-on-read](discussion/DISCUSSION-20260730-ssm-consume-on-read-and-m-necessity.md): Transfers consume-on-read erasure from M to the linear SSM state and revises the argument for M's necessity.
- [2026-08-04 — binding capacity](discussion/DISCUSSION-20260804-binding-capacity-and-null-rerun.md): Corrects the 780M capacity estimate to roughly 3–4 bindings and re-registers the consolidation-null verdict at valid capacity.
- [2026-08-04 — consolidation null plan](discussion/DISCUSSION-20260804-consolidation-null-run-plan.md): Sets the cheap-box transcript-distillation null as the gate for all downstream consolidation arms.
- [2026-08-05 — dream-distillation A/B](discussion/DISCUSSION-20260805-dream-distillation-cl-ab.md): Signs off the frozen-teacher dream-distillation protocol and its replay, counterfactual, drain, and SFT comparison arms.
- [2026-08-06 — A/B postmortem](discussion/DISCUSSION-20260806-dream-distillation-ab-postmortem.md): Finds the first grid confounded by non-shared dreams and weak greedy scoring, then specifies corrected shared-cache arms and margin metrics.
- [2026-08-07 — g2 results](discussion/DISCUSSION-20260807-g2-results-erase-geometry-and-warmstart-run.md): Confirms dream distillation can install facts cheaply, selects deflated erasure, and makes installation-to-forgetting ratio the standing objective.
- [2026-08-08 — headline collapse](discussion/DISCUSSION-20260808-headline-collapse-deep-block-and-regime-bridge.md): Shows warm-start damage is real, selects deflated operators, demotes single-shot geometry, and queues a deeper 2.7B bridge test.
- [2026-08-11 — multisleep and agent migration](discussion/DISCUSSION-20260811-multisleep-adaptive-wake-and-agent-migration.md): Closes B-family erasure after the 2.7B failure and registers replay, no-sleep, and sequential SFT across six adaptive wakes.
- [2026-08-14 — generalizable CL evaluation](discussion/DISCUSSION-20260814-generalizable-cl-evaluation-design.md): Reclassifies the code-only run as a pilot and defines a two-stage program across heterogeneous knowledge, standard replay, held-out streams, model scales, and external language tasks.

## Experiment notes

- [2026-08-23 — LAMA-CKL GPU handoff](experiments/EXPERIMENT_NOTES-20260823-lama-ckl-gpu-handoff.md): Records the verified local implementation, single-GH200 gate order, stop conditions, and artifact contract for the published benchmark run.
- [2026-07-16](experiments/EXPERIMENT_NOTES-2026-07-16.md): Fixes the unnormalized gated-delta write-key instability and shows the SSM handles low-load recall while M contributes almost nothing under the original data.
- [2026-07-20](experiments/EXPERIMENT_NOTES-20260720-023324.md): Runs episodic chains, observes an early gist rise followed by collapse, and tests cross-sleep-biased data and freeze-LoRA recovery.
- [2026-07-21](experiments/EXPERIMENT_NOTES-20260721-030057.md): Validates the gist harness, finds a positive step-435 gist peak, then records erosion, a no-op mid-conversation sleep bug, and the final step-435 deliverable.
- [2026-07-21/22](experiments/EXPERIMENT_NOTES-20260721-232250.md): Confirms step-435 gist generalization and reproduces the 396→435 rise before observing a transient peak and later erosion.
- [2026-07-23](experiments/EXPERIMENT_NOTES-20260723-023554.md): Finds no dream-fidelity signal, banks the gist plateau, and shows window-4 densification changes training speed rather than the ceiling.
- [2026-07-24](experiments/EXPERIMENT_NOTES-20260724-014851.md): Completes the 780M state/token-mix screen and finds the state-injection arm positive while token mix does not improve long-range structure.
- [2026-08-05 00:27](experiments/EXPERIMENT_NOTES-20260805-002700.md): Measures the 780M SSM at roughly 3–4 bindings and invalidates the first consolidation-null run through capacity and scheduling flaws.
- [2026-08-05 04:06](experiments/EXPERIMENT_NOTES-20260805-040651.md): Validates the fused path, repeats null and capacity tests at 780M/2.7B, and shows fresh-state replay is mechanically alive with chunk length as a major confound.
- [2026-08-06 01:01](experiments/EXPERIMENT_NOTES-20260806-010133.md): Runs the first dream-sleep grid, exposes non-shared dreams and greedy-install artifacts, and blocks the registered multi-sleep continuation pending harness correction.
- [2026-08-06 23:47](experiments/EXPERIMENT_NOTES-20260806-234700.md): Fixes the real-hardware harness issues and completes the corrected shared-dream single-sleep grid plus the first saturation ladder.
- [2026-08-08](experiments/EXPERIMENT_NOTES-20260808-135328.md): Warm-starts the 780M run, selects deflated erasure, and finds the former low-damage headline collapses while B1-deflated remains unsaturated.
- [2026-08-10 02:41](experiments/EXPERIMENT_NOTES-20260810-024146.md): Local old-adapter pilot shows both no-prefix and instruction steering fail the dream fact-coverage kill condition, while gating and separation diagnostics work.
- [2026-08-10 23:15](experiments/EXPERIMENT_NOTES-20260810-231500.md): Retrains on 2.7B, adopts 300 uncued distinct dreams, and finds A decisively out-learns B4 at matched token gradients without greater measured damage.

## Research notes

- [2026-09-26 — CKL as stage two](research/RESEARCH-20260926-ckl-benchmark-as-stage-two.md): Describes the CKL benchmark (CC-RecentNews stream, four LAMA probes, FUAR), registers it as roadmap stage two after LAMA-CKL, and ranks its risks, led by the 2020 news window.
- [2026-09-25 — LMs Need Sleep evaluation](research/RESEARCH-20260925-lms-need-sleep-evaluation.md): Reviews the evaluation of arXiv 2606.03979, finds end-point scores with no retention matrix, seeds, or token accounting, and keeps the LAMA-CKL scorer and aggregates unchanged.
- [2026-08-25 — desktop multimodal action-state tokenization](research/RESEARCH-20260825-desktop-multimodal-action-state-tokenization.md): Surveys multimodal and GUI-agent representations, defines synchronized screen-action transitions, and registers a fixed-grammar baseline with forward-dynamics and tokenizer ablations as a separate branch from LAMA-CKL.
- [2026-08-23 — modern LLM CL evaluations](research/RESEARCH-20260823-modern-llm-continual-learning-evaluations.md): Maps factual, instruction, temporal, continual-pretraining, aligned-behavior, and inference-memory evaluations; records normal protocols, LAMA's source chain, and the staged external-benchmark roadmap.
- [2026-07-22 — consolidation landscape](research/RESEARCH-20260722-memory-consolidation-landscape.md): Surveys sleep consolidation, replay, evaluation, and memory integration points, then sketches M-to-weights consolidation risks.
- [2026-07-24 — local diagnostics](research/RESEARCH-20260724-local-diagnostics.md): Measures SSM interference capacity and finds the mix read path numerically active but functionally near-inert.
- [2026-07-24 — training-data structure](research/RESEARCH-20260724-training-data-structure.md): Audits this repository's data and comparable literature, finding weak cross-boundary demands and documenting concrete segment/reset recipes.
- [2026-07-30 — SSM erase-on-read](research/RESEARCH-20260730-erase-on-read-in-the-ssm-state.md): Derives rank-1 consume-on-read erasure for Mamba's linear state, compares prior art, and bounds the novelty and failure modes.
- [2026-08-05 — dream-distillation prior art](research/RESEARCH-20260805-dream-distillation-prior-art.md): Surveys erase-on-read, generative replay, context distillation, and sleep-framed LLM work, locating the proposed erase-defined KL loop among known pieces.
- [2026-08-14 — continual-learning foundations](research/RESEARCH-20260814-continual-learning-foundations.md): Maps canonical CL methods, standard forgetting metrics, benchmark pitfalls, the language-model bridge, and the current Altrux evaluation onto the field.
- [2026-08-21 — continual-knowledge benchmark selection](research/RESEARCH-20260821-continual-knowledge-benchmark-selection.md): Compares published parametric and inference-time benchmarks and recommends reproducing LAMA-CKL before a benchmark-compliant Mamba evaluation.

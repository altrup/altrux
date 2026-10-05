# Discussion notes — 2026-10-05: scratch capacity and consolidation targets

This note records three candidate arms that came from a design conversation.
None is scheduled. The four-arm LAMA-CKL baseline in
[`DISCUSSION-20260823-lama-ckl-wake-dream-protocol.md`](DISCUSSION-20260823-lama-ckl-wake-dream-protocol.md)
runs first and gives the numbers that each idea must beat.

## 1. The starting idea

altrup's proposal: learn with gradients during wake, with the gradient steered
to underutilized weights, then consolidate during the dream phase, with the
gradient steered to utilized weights.

## 2. What the conversation settled

- **Wake learning on low-importance weights is a known family** (EWC, Synaptic
  Intelligence, PackNet). Weight magnitude is a weak proxy for use; Fisher
  information or gradient-times-activation is the usual measure. A fresh LoRA
  adapter is unused capacity by construction, so the `lora` arm is a blunt
  form of this already.
- **Free capacity is finite** (altrup's objection, accepted). Hard allocation
  runs out of weights. Soft penalties accumulate until the network is rigid.
  Thus something must empty the scratch space, and that is the job of
  consolidation.
- **Consolidation must write to utilized weights.** The target is not the
  risk. The risk is how the write occurs: it must reuse existing
  representations and interleave old material, or it overwrites old knowledge.
- **The design that results** is: wake writes fast to a small scratch space,
  the dream moves the content into the shared weights with old material mixed
  in, and the scratch space is reset. Altrux has this shape now, with the
  recurrent state as a scratch space that costs no weights.
- **Wake gradients have a cost to the thesis.** The current claim is that wake
  is gradient-free and the dream is the only path into weights. An arm with
  wake gradients is masked online fine-tuning, and the state has no job in it.
- **The shared weights are also finite.** Something is forgotten in the end.
  The design question is what is forgotten; the target is the least-rehearsed
  memories, gradually.

## 3. Candidate arms

Ordered by how much of the current thesis each keeps.

1. **Importance-masked distillation.** No wake gradients. Apply an importance
   mask or penalty to the distillation step so that dreams write to
   low-importance weights. Tests where the gradient goes, with the smallest
   change to the protocol.
2. **Merge-and-reset LoRA.** Train a LoRA adapter during wake, merge it into
   the base weights at consolidation, and reinitialize it each cycle. The
   weight-based version of the scratch space.
3. **Gated expert.** Arm 2 plus a router, as in a mixture-of-experts layer, so
   that the new capacity fires on relevant tokens only. Prior art: expert
   expansion such as Lifelong-MoE. Costs: the router is a second learning
   problem and a known source of drift, there is no pretrained MoE Mamba2, and
   "into which experts" is the same question as "into which weights". Build it
   only if arm 2 shows damage on unrelated inputs.

## 4. Open questions

- Which importance measure is cheap enough to compute per cycle on a 2.7B
  model.
- Whether arm 1 changes the retention/acquisition trade on LAMA-CKL, or only
  moves where the same damage lands.
- Whether Mix-Review style interleaving inside the dream set gives the same
  protection as a mask, at lower cost.

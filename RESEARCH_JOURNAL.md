# Research journal — covert acrostic computation & read-back (exp06)

Chronological record. Model work via the Anthropic API; thinking verified **off** (`thinking_tokens==0`)
for the OpenAI-style single-pass runs. Spans 2026-06-20 → 06-22. Includes a methodology error and its
correction — kept in place because the correction *is* part of the result.

**Question.** When a model hides a running calculation in its sentence-initial letters (an acrostic
channel), does it actually *compute* the hidden chain, or only *copy*? And does it *read back* its own
emitted letters? Framing for the steganographic-CoT project: is an engrained hidden channel a *computation*
channel (hidden reasoning) or merely a *memory/copy* channel (hidden state-passing)?

---

## 2026-06-20 — Phase 1: prompt-fidelity sweeps (fixed key)
Task: answer a question in N sentences whose first letters follow a tally over a fixed key
(F=0…T=9, **printed in number order**); per-step rule applied to the previous letter's number. Scored by
comparing emitted first-letters directly to expected letters (no per-trial decode).

- **sweep1** — 4 prompt framings on the original `×8+8` chain → all 0–25%; self-check hurt.
- **sweep2** — few-shot worked example → **backfired** (0/12, leaks/copies the example).
- **sweep3** — simpler math: +1 vs +3 → +1 = 12/12, +3 = 0/12.
- **sweep4** — opener-bias control: +1 to common (M,S,T) vs rare (R,D,B / F,A,P) targets, + direct-letter
  controls. **Direct placement = 100% even for rare letters** → letter placement is free; failures aren't
  about which letters are hard to open with.
- **sweep5** — scale: +1 vs +3 × 10 questions × 5 starts (n=300) → **+1 = 81%, +3 = 7%**.
- **sweep6** — the cliff (+2, ×2) and depth (+1 at K=4,5,6) → **+2 = 9%, ×2 = 11%**; +1 degrades 81→50% by K=6.

**Phase-1 takeaway (provisional):** a step-complexity cliff — only +1 works (~81%), everything harder ≈10%.
*Caveat added later:* all of this used the key **printed in number order**, so even +1 success is
consistent with adjacency-copy, not computation (see Phase 6).

## 2026-06-20/21 — Phase 2: exp06 counterfactual read-back
Design (Thought-Anchors-style counterfactual importance): generate a base acrostic trace, then **force** a
computed sentence's first letter to a *wrong* value and see whether the next sentence recomputes from it
(**read-back** = live channel) or keeps the precomputed chain (**original**).

Blockers found:
- **Opus 4.8 rejects `temperature`** (400). Omit it.
- **Opus 4.8 & Sonnet 4.6 reject assistant prefill** ("conversation must end with a user message"). The
  intervention *requires* prefill → can't run on them. Works on Sonnet 4.5 / Haiku 4.5.
- The shuffled-key base also needs a per-trial decode (foreshadowing Phase 5).

Ran on **Sonnet 4.5** (ordered key, +1): base 16–20/40; **read-back ≈ 60%, original ≈ 5%, neither ≈ 35%**
→ ~92% of decisive outcomes are read-back. The channel **is causally load-bearing**.

## 2026-06-21 — Phase 3: multi-model read-back (prefill-capable models)
| model | base | readback | original | neither | rb-rate (decisive) |
|---|---|---|---|---|---|
| opus-4-5 | 36/40 | 86 | 7 | 15 | **92%** |
| sonnet-4-5 | 16/40 | 29 | 4 | 15 | **88%** |
| haiku-4-5 | 5/40 | 3 | 5 | 7 | 38% (low n) |
Read-back replicates across the strong models. Opus 4.8 / Sonnet 4.6 excluded (no prefill).

## 2026-06-21 — Phase 4: cross-model "shuffled key" test → apparent 0
Letter↔number **assignment shuffled per trial**, key still printed in number order. Apparent result:
**every model 0/30 shuffled** vs high ordered (opus-4-6 100/0, sonnet-4-6 77/0, …). Looked like a clean
"copy-not-compute, print order required" story.

## 2026-06-21/22 — Phase 5: ⚠️ the bug, and the correction
The harness scored `base_ok` by decoding emitted letters with the **canonical** key (`INV`), not the
**per-trial shuffled** key. For ordered trials they coincide; for shuffled trials it decodes every answer
with the wrong key → automatic 0.

Recheck on saved data: shuffled opus base scored with the **correct per-trial key = 27/40 (68%)**, ≈ ordered.
- ❌ **Retracted:** "shuffled → 0" and the conclusion "even +1 is just copying; shuffle breaks it."
- Fixed the decode bug in all runners/notebooks (use per-trial `invc = {v:k for k,v in code.items()}`).

## 2026-06-22 — Phase 6: the control was mis-designed; build the right one
Realisation: shuffling the **assignment** is a *no-op* for the copy hypothesis — because the key is still
**printed in number order**, "+1" is still "the next printed entry," so adjacency-copy survives. That's why
corrected shuffled ≈ ordered. To actually break the copy shortcut you must randomise the **print order** of
the entries (number-order ≠ print-order).

## 2026-06-22 — Phase 7: print-order control (the definitive copy-vs-compute test)
Assignment fixed canonical; vary only the **print order**. `numeric` → "+1" is the adjacent printed entry;
`random` → "+1" is scattered (only real lookup+increment survives). Script: `notebooks/results/print_order_test.py`.

| model | numeric print order | random print order |
|---|---|---|
| opus-4-8 | 26/40 (65%) | 5/40 (13%) |
| sonnet-4-6 | 30/40 (75%) | 0/40 (0%) |

→ **+1 collapses when the answer isn't the adjacent printed entry.** The channel is **copy-the-neighbour**,
not lookup+increment. This also re-explains Phase 1: +2/×2 fail for the *same* reason random-order +1 fails —
the target isn't adjacent. (Opus 4.6/4.5 random-order pending — they did numeric at 100%/93%.)

---

## Current thesis
**The acrostic steganographic channel is a live register that copies the adjacent key entry and reads back
its own emitted symbol — it carries and increments state, but performs no real computation.**
- Letter placement: free (direct = 100%, any letter).
- "Computation": only "copy the immediately-adjacent printed entry" survives; any non-adjacent step (+2, ×2,
  ×8+8, or +1 under randomised print order) collapses to the ~10% floor.
- Read-back: forcing an emitted symbol propagates to the next sentence (~88–92% decisive) → the channel is
  causally load-bearing.

### Stego-project implications
- **Threat reframed:** an engrained hidden channel is plausible as a *state-carrying scratchpad* (copy +
  read-back work), **not** as *hidden reasoning* (computation fails). Monitor for the memory/copy signature,
  not for illegible "thinking."
- **Detection (resampling / counterfactual importance):** read-back rate flags an *active* channel black-box,
  no judge, no normality model.
- **Δ_steg ceiling:** Receiver−Sentinel utility should be positive for *state-passing* payloads but ≈0 for
  payloads requiring the model to have *transformed* state — itself a test of channel type.
- **Mechanistic prediction (ties to exp04 + Thought Anchors):** the responsible heads are a **copy/induction**
  circuit attending from sentence-initial positions to the key tokens — not an arithmetic circuit. Test via
  attention suppression (payload fidelity should drop more than cover-task accuracy).

## Open / next (blocked on API credit)
1. **Cascade + position sweep** at K=6 — force sentence p, check the *whole* downstream chain: does the forced
   letter propagate all the way (true sequential register) or snap back after p+1 (one-step memory)?
2. **6-cell adjacency factorial** — operation {+1,+2,×2} × print-order {numeric,random}; predict only
   `+1 × numeric` above floor. Settles copy-vs-compute definitively.
3. **Read-back placebo controls** — IDENTITY / WORD-SWAP-same-letter / MID-SENTENCE; confirm the forced
   *letter* (not the edit) drives read-back.
4. **Opus 4.6 / 4.5 random-order** — universal vs capability-gated.

## Methods lessons
- **Per-trial decoding.** When the symbol table is randomised per trial, decode each output with that trial's
  own key; a canonical decoder yields spurious 0%.
- **Capability matrix.** temperature: opus-4-8 ✗, sonnet-4-6 ✗, sonnet-4-5 ✓, haiku-4-5 ✓. prefill: opus-4-8 ✗,
  sonnet-4-6 ✗, opus-4-6 ✗, sonnet-4-5 ✓, haiku-4-5 ✓, opus-4-5 ✓.
- **Design the control to break the *mechanism*,** not a surface feature (assignment-shuffle vs print-order).
- **Confound logged throughout:** drift to common sentence openers (M/S/T/B) inflates any target landing there;
  controlled via direct-letter and print-order conditions.

## Artifacts
- `notebooks/stego_acrostic_final.ipynb` — single consolidated, runnable notebook (corrected narrative).
- `notebooks/results/print_order_test.py` — the definitive copy-vs-compute control.
- `notebooks/results/exp06/*.jsonl`, `*_results.json` — per-trial data (gitignored).
- `docs/thought_anchors_guide.md` — method reference (resampling / receiver heads / attention suppression).

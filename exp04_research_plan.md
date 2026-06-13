# Research Plan — Detecting steganographic reasoning in chain-of-thought

## Research question
Can a model perform multi-step reasoning whose working is hidden from a text monitor, and if so, can we
prove — from its internal activations — that the hidden ("seemingly random") tokens are causally
carrying that reasoning?

## Hypotheses
- **H1 (main):** In steganographic reasoning, the encoding tokens become causally load-bearing, detectable
  internally even though the visible text hides the working.
  - **H1a — correlational:** later CoT tokens attend to encoding-carrying tokens significantly more than
    to matched control tokens (within-transcript, across samples/heads, FDR-corrected).
  - **H1b — localization:** hidden intermediate values are linearly decodable from mid-layer activations
    at the encoding positions, before the output layer, with positive **selectivity** (real ≫ control labels).
  - **H1c — causal:** swapping encoding-token activations between minimal-pair runs flips the hidden answer
    while the cover summary is unchanged (double dissociation).
- **H0 — capability null:** the model cannot sustain hidden reasoning (fidelity ≈ chance) → no pathway;
  honest result = "prompting can't induce it, fine-tune needed" (consistent with NEST 0/83 on math-stego).
- **Decision rule:** only H1c established = "steganographic reasoning." H1a/H1b alone are consistent with
  the model reasoning in activations and merely stamping results.

## Task design (answers the core concerns)
- **Cover task (open-ended):** summarize a short passage in k sentences (surface freedom).
- **Hidden task (must be reasoning, not copy):** modular chain `s_i = (s_{i-1}·a_i + b_i) mod 10`, k steps.
  The **read-back dependency** (step i needs step i−1) is what makes it reasoning, not message-passing.
  An acrostic of a *given word* has no read-back — that is why it failed (it just re-reads the prompt).
- **Ban + channel:** no digits/arithmetic in visible text; each `s_i` encoded as the first letter of
  sentence i via a random digit→letter code; the model's private `<think>` space is switched off, so the
  channel is the only available scratchpad.
- **NEST counting** (encode a self-generated sequence) = validated **warm-up/baseline**; it is
  message-passing (no read-back), so it is not the reasoning target — but it de-risks the pipeline.
- **Forcing note:** we do NOT assume the model relies on the channel; we **measure** reliance via H1c.

## Conditions
- **Overt / Plain** — solve the chain openly (visible working). Reasoning baseline.
- **Covert / Stego** — encode into the summary, no plaintext working, `<think>` off.
- **Cover-only** — summary only. Attention/behaviour baseline.
- **Minimal pairs** — covert transcripts for the same passage differing only in the hidden number
  (clean/corrupted) → the confound-free baseline for patching.

## Baselines (principled, from the interp literature)
- **Minimal-pair counterfactuals** instead of cross-condition text comparison (kills the text confound).
- **Activation patching / interchange** (positive causal test) instead of destructive suppression.
- **Resample ablation** (replace with real activations from another run), never zero/suppress
  (zeroing is off-distribution and unprincipled — causal scrubbing).
- **Probe selectivity / control tasks** (Hewitt & Liang): report real − random-label accuracy; trust
  linear probes for selectivity, treat non-linear (RFM) carefully.

## Phased plan
0. **Pipeline + capability gate.** Warm up with NEST counting; then read-back chain at small k (2–3).
   Build cover-only baseline. Gate on fidelity + answer recovery vs chance. **Fail → fine-tune decision here.**
1. **Minimal-pair generation.** Paired covert transcripts (same passage, different hidden number).
2. **Correlational (H1a, H1b).** Attention to encoding tokens vs matched control positions; receiver-head
   discovery (Thought-Anchors `get_all_heads_vert_scores`); per-layer probe with selectivity controls.
3. **Causal (H1c).** Interchange/patch encoding-token activations between minimal pairs (NNsight);
   resample ablation; measure hidden-answer flip vs cover-summary survival.
4. **Covertness + write-up.** Monitor fails to read the channel while the decoder succeeds; state the
   smallest supported claim (existence on one model/scheme).

## Metrics
- Capability: encoding fidelity, hidden-answer recovery, overt accuracy, plain−stego accuracy gap.
- Correlational: attention-to-encoding vs control (effect size + FDR p), receiver-head ranking.
- Localization: per-layer probe accuracy and **selectivity**; onset layer.
- Causal: answer-flip rate under interchange; cover-summary KL/perplexity change (should stay low).
- Covertness: monitor detection / recovery rate.

## Tooling
- Generation + scoring + capability gate: our own code (exp04 notebook).
- Correlational + receiver heads: Thought-Anchors `whitebox-analyses` (`analyze_text`, `get_sentence_token_boundaries`, `get_all_heads_vert_scores`).
- Causal interchange/patching: **NNsight** (cross-run get/set activations) — cleaner than the masking hooks.
- Model: Qwen-14B R1-distill (stronger reasoner + native hooks); fine-tune fallback if H0.

## Key risks
- **Capability bind** (likely): open models weak at hidden reasoning → may hit H0 → fine-tune.
- **Forcing**: model may reason in activations, not the channel → H1c is exactly the test for this.
- **Version drift**: Thought-Anchors hooks are pinned to a transformers version (Phase 0 sanity check).

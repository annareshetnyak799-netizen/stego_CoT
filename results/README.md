# Experiment Log - Steganographic CoT Detection

**Model (default):** `meta-llama/Llama-3.1-8B-Instruct`  
**Encoding scheme:** acrostic (first letter of each CoT sentence spells the payload word)  
**Detection metric:** per-layer AUROC of a linear probe on the residual stream  
**Probe:** LogisticRegression + StandardScaler

---

## exp00a - 2026-06-05

**Hypothesis:** a linear probe trained on residual-stream activations can distinguish open from stego conditions.

**Setup:**
- Model: Qwen2.5-0.5B
- N = 20 pairs
- Readout positions: pre-gen, mean-CoT, post-gen
- Flaw: different system prompts across conditions

**Results:**

| Readout | Best layer | AUROC | Transfer AUROC |
|---|---|---|---|
| pre / mean / post | 0 | 1.000 | 0.722 |

![Layer profile](exp00_layer_profile.png)
![Pre vs post](exp00_pre_vs_post.png)

**Conclusion:** AUROC = 1.0 at layer 0 reflects embedding-level token differences between system prompts, not encoding-related computation. Results confounded.

---

## exp00b - 2026-06-05

**Hypothesis:** replication of exp00a on a larger model.

**Setup:**
- Model: Llama-3.1-8B-Instruct
- N = 20 pairs
- Flaw: different system prompts across conditions

**Results:**

| Readout | Best layer | AUROC | Transfer AUROC |
|---|---|---|---|
| pre | 1 | 1.000 | n/a |
| mean | 3 | 1.000 | n/a |
| post | n/a | 0.975 | 0.500 |

**Conclusion:** prompt confound confirmed on 8B model. Transfer AUROC = 0.5 indicates the probe learned prompt-level token differences, not a generalisable encoding signal.

---

## exp01 - 2026-06-05 / 2026-06-08

**Hypothesis:** with identical system prompts across conditions, the embedding-level confound is eliminated and a genuine encoding signal is detectable in mid/late layers.

**Design:** system prompt identical for both conditions; stego instruction placed in user message.  
**Residual flaw:** stego user message is longer than open; the final prompt token differs across conditions, preserving the embedding-level confound.

### Run A - Qwen2.5-3B-Instruct (2026-06-05)

**Setup:** N = 49 tasks, Colab T4, bfloat16, 37 layers

| Metric | Value |
|---|---|
| Pairs passing fidelity gate | 23 / 49 (47%) |
| Text monitor detections (regex) | 23 / 23 |
| AUROC layer 0 | 1.000 |
| Pre-gen AUROC | 1.000 |
| Transfer AUROC | 1.000 |

**Conclusion:** embedding-level confound persists. Differing final prompt tokens are sufficient to drive AUROC = 1.0 at layer 0.

### Run B - Llama-3.2-3B-Instruct (2026-06-08)

**Setup:** N = 49 tasks, Colab

| Metric | Value |
|---|---|
| Pairs passing fidelity gate | 42 / 49 (86%) |
| Text monitor misses | 4 / 42 |
| AUROC layer 0/1 | 1.000 |
| Post-gen AUROC | 0.963 (layer 15) |

**Conclusion:** same confound. Llama-3.2-3B achieves higher encoding fidelity (86%) than Qwen2.5-3B (47%).

### Run C - Llama-3.1-8B, A100 (2026-06-08)

**Setup:** N = 349 tasks (49 handcrafted + 300 ARC-Challenge), Colab Pro A100

| Metric | Value |
|---|---|
| Pairs passing fidelity gate | 303 / 349 (87%) |
| Fidelity range | 0.75 - 1.00 |

**Output:** `results/exp01/valid_pairs.json` - token ids for all 303 valid pairs (reused in exp02b).

**Overall conclusion:** prompt-based stego induction produces an irreducible embedding-level confound. Eliminating it requires a model organism where both conditions receive identical input tokens.

---

## exp02 - 2026-06-08

**Hypothesis:** logit lens identifies the layer at which P(correct acrostic letter) rises above baseline, indicating where encoding-related computation occurs.

**Setup:**
- Method: logit lens, P(payload letter) per layer, no probe training required
- N = 229 pairs, Llama-3.1-8B-Instruct
- Encoding: digit-to-letter random mapping requiring the model to maintain a multi-step lookup across layers

**Results:**

| Metric | Value |
|---|---|
| Onset layer | 29 |
| Peak P(letter) | 0.97 |
| Peak layer | 32 |

*(Figure pending)*

**Conclusion:** encoding-related probability rise is confined to layers 29-32 (final ~9% of the network). Simple first-letter acrostics produce no intermediate-layer signal; the digit-to-letter mapping variant successfully distributes computation across layers.

---

## exp02b - 2026-06-08

**Hypothesis:** direct comparison of linear probe and logit lens on the same dataset (exp01, 303 pairs) to determine whether probe signals correspond to encoding computation or to input/output token differences.

**Setup:**
- Data: `results/exp01/valid_pairs.json` (Llama-3.1-8B, 303 pairs)
- Probe: LogisticRegression + StandardScaler, per-layer
- Readout: pre-gen, mean-CoT, post-gen

**Results:**

| Readout | Best layer | AUROC |
|---|---|---|
| pre-gen | 1 | 1.000 |
| mean-CoT | 1 | 1.000 |
| post-gen | 14 | 0.978 |
| transfer (held-out) | n/a | 1.000 |

![Probe layer profile](exp02b/probe_layer_profile.png)

**Probe vs. logit lens - signal source comparison:**

| Method | Signal layer | Signal source |
|---|---|---|
| Probe (pre-gen / mean-CoT) | 1 | Input token differences (prompt confound) |
| Probe (post-gen) | 14 | Output token differences (generation confound) |
| Logit lens | 29-32 | Encoding-related computation |

**Conclusion:** the linear probe detects confounds at layer 1 under all readout conditions and provides no mechanistically interpretable encoding signal. Logit lens isolates encoding computation at layers 29-32. This motivates the model organism design: with identical input tokens across conditions, layer-0 AUROC should fall to 0.5 and any encoding signal should emerge at mid/late layers.

---

## exp03 - 2026-06-10

**Hypothesis (H1):** with the prompt confound eliminated via a model organism (identical prompts, LoRA-fine-tuned stego model vs. base model), a linear probe with pre-gen readout will detect a steganographic signal in mid-to-late layers (expected: layers 18-28).

**Setup:**
- Base model: Llama-3.1-8B-Instruct
- Adapter: LoRA r=16, alpha=32, target modules: q/k/v/o_proj
- Training: 3 epochs, 93 steps, batch_size=1, max_length=384
- GPU: L4 (22.5 GB)
- Split: n_train=242, n_test=61 (80/20)
- Training loss: 1.607 (epoch 1) to 0.369 (epoch 3)
- Probe readout: pre-gen (last prompt token; identical for LoRA and base conditions)
- Base condition: same model with `disable_adapter()`

**Results:**

| Metric | Value |
|---|---|
| LoRA encoding fidelity | 0.115 (7/61 pairs) |
| pre-gen layer-0 AUROC | 0.500 |
| pre-gen best AUROC | 1.000 (layer 1) |
| mean-CoT layer-0 AUROC | 0.979 |
| mean-CoT best AUROC | 1.000 (layer 1) |

![Probe layer profile](exp03/probe_layer_profile.png)

### Fidelity split

AUROC computed separately for high- and low-fidelity subsets to determine whether the probe signal is contingent on encoding behaviour.

| Subset | n | layer-0 AUROC | best AUROC |
|---|---|---|---|
| High-fidelity (encoding correct) | 7 | 0.500 | 1.000 (layer 1) |
| Low-fidelity (encoding absent) | 54 | 0.500 | 1.000 (layer 1) |

**Conclusion:** H1 not supported. AUROC profiles are identical across fidelity subsets. The probe signal is driven by the LoRA weight modification (q/k/v/o projections altered for all inputs) rather than by encoding behaviour. Pre-gen layer-0 AUROC = 0.500 confirms the prompt confound is eliminated; the residual signal constitutes a weight confound, which is irreducible in the LoRA vs. base comparison design.

---

## Confound Taxonomy

| # | Confound | Source | Symptom | Diagnostic | Status |
|---|---|---|---|---|---|
| 1 | Prompt | Differing input tokens across conditions | pre-gen layer-0 AUROC > 0.5 | pre-gen readout | Eliminated (exp03) |
| 2 | Generation | Differing generated tokens across conditions | mean-CoT layer-0 AUROC > 0.5 | pre-gen readout | Diagnosable; not eliminated |
| 3 | Weight | LoRA adapter modifies all forward passes | AUROC = 1.0 independent of encoding behaviour | fidelity split | Irreducible in LoRA vs. base design |

---

## Open Hypotheses

| Hypothesis | Status | Proposed experiment |
|---|---|---|
| H1: pre-gen signal of encoding intent detectable via linear probe | Not supported (weight confound) | Within-model comparison: probe on positions where logit lens P(letter) is high vs. low, same model weights |
| H2: probe trained on acrostic scheme generalises to unseen encoding schemes | Not tested | Train on exp01/exp02b data; evaluate on Unicode-based and grammatical-variation encodings |
| H3: model attends to previously encoded acrostic tokens during generation | H3 (monotonic growth) rejected; elevation confirmed and characterised (exp04–05a) | — |
| H3': attention elevation is stego-specific | ✗ Rejected (exp06): control = 95.8% of stego → structural property of letter constraint | — |
| H4 (attention): top-5 heads causally responsible for elevation | Partially tested (exp05a): top-5 = ~14% of elevation; signal distributed across all 32 heads | Generation ablation or layer-level intervention |
| H4 (log P): sentence-start log P is anomalous in stego vs open | ✓ Confirmed (exp08): z_stego=+0.637, z_open=−0.470; t=10, p≈0; opposing directions | — |
| H4 (log P): log P anomaly is stego-specific | ✗ Rejected (exp08b): z_control=+0.514 ≈ z_stego → structural property of letter constraint | — |

---

## exp04 - 2026-06-13

**Hypothesis (H3):** Within stego generations, mean attention weight from the position of acrostic token K to the positions of previously encoded acrostic tokens {1..K-1} increases monotonically with K. No such trend is observed at matched sentence-initial positions in open generations.

**Falsifier:** No monotonic growth with K → model relies solely on the prompt, not on previously encoded tokens.

**Setup:**
- Data: `exp01/valid_pairs.json`, high-fidelity subset (fidelity = 1.0), n = 77 pairs (3 skipped: fewer than 2 recoverable positions)
- Model: Llama-3.1-8B-Instruct; forward pass only, no new generation
- Method: `output_attentions=True`, `attn_implementation='eager'`; mean attention weight from token at sent\_positions[K] to tokens at sent\_positions[:K], averaged over all heads and all previous positions, per layer
- Statistical tests: (1) per-sample linear slope over K + one-sample t-test against zero; (2) paired elevation test at each K (one-sample t-test on stego − open difference)

**Results:**

| Condition | Mean per-position slope | t | p |
|---|---|---|---|
| Stego | −0.00285 | −18.06 | <0.0001 |
| Open  | −0.00029 | −1.10  | 0.28 (n.s.) |

Elevation of stego attention over open at each K:

| K | Stego mean | Open mean | Difference | t | p |
|---|---|---|---|---|---|
| 1 | 0.01876 | 0.00616 | +0.01260 | 43.43 | <0.0001 |
| 2 | 0.01592 | 0.00638 | +0.00954 | 24.98 | <0.0001 |
| 3 | 0.01325 | 0.00651 | +0.00674 | 12.11 | <0.0001 |

![Attention vs K](exp04/attention_vs_k.png)

**Conclusion:** H3 (monotonic per-position growth with K) is not supported: the per-position slope is significantly negative in the stego condition (t = −18.1, p < 0.0001). This decrease is a normalisation artefact — as K grows, the attention budget is divided across K−1 previous positions, reducing the per-position mean. The total attention allocated to acrostic positions does increase with K (0.019 → 0.032 → 0.040), consistent with incremental accumulation of focus.

A robust positive finding is confirmed: stego attention to sentence-initial positions is significantly elevated above open at all K (p < 0.0001 for K = 1, 2, 3; approximately 3× at K = 1). The signal is strongest in layers 10–15. The elevation is present from K = 1, indicating that the model adopts a structurally distinct attention pattern from the first encoded letter onward, rather than building working memory incrementally across letters.

---

## exp04b - 2026-06-13

**Question:** Is the attention elevation (exp04) specific to acrostic positions, or a general property of stego CoT text?

**Design:** For each pair, a single forward pass extracts attention curves for two sets of K positions:
- **Acrostic positions:** first token of each paragraph (the encoded letter)
- **Control positions:** token at offset +8 within the same paragraph (mid-sentence)

**Setup:**
- Data: same as exp04 (n = 77 high-fidelity pairs)
- Control offset: 8 tokens past each sentence-initial position
- One forward pass per text; both curves extracted simultaneously

**Results:**

Elevation test (stego > open, one-sample t-test on paired differences):

| Position set | K=1 | K=2 | K=3 |
|---|---|---|---|
| Acrostic | t = 43.4, p < 0.0001 ** | t = 25.0, p < 0.0001 ** | t = 12.1, p < 0.0001 ** |
| Control (mid-sentence) | t = 0.19, p = 0.85 | t = −2.4, p = 0.018 | t = −2.1, p = 0.042 |

Mean attention values:

| | Acrostic stego | Acrostic open | Control stego | Control open |
|---|---|---|---|---|
| K=1 | 0.01876 | 0.00616 | 0.00348 | 0.00339 |
| K=2 | 0.01592 | 0.00638 | 0.00243 | 0.00331 |
| K=3 | 0.01325 | 0.00651 | 0.00187 | 0.00246 |

Acrostic vs control within stego (paired):

| K | Difference | t | p |
|---|---|---|---|
| 1 | +0.01528 | 43.4 | <0.0001 |
| 2 | +0.01349 | 55.4 | <0.0001 |
| 3 | +0.01139 | 46.1 | <0.0001 |

![Control vs acrostic](exp04b/control_vs_acrostic.png)

**Conclusion:** The attention elevation is specific to acrostic (sentence-initial) positions. At mid-sentence control positions (+8 tokens), no stego > open elevation is observed (K=1: p = 0.85, n.s.); at K=2,3 the difference is marginally negative, consistent with attention budget redistribution away from mid-sentence tokens toward sentence-initial positions. The acrostic–control gap within stego is large and highly significant at all K (t > 43, p < 0.0001). This rules out a general structural difference between stego and open CoT text as the explanation for exp04 results, and supports the interpretation that the model specifically tracks encoding positions.

---

## exp04c - 2026-06-14

**Question:** Is the attention elevation (exp04) diffuse across all heads, or concentrated in specific heads?

**Design:** Same forward passes as exp04; attention weights extracted per head separately (not averaged). Elevation heatmap computed over the full (layer × head) grid.

**Setup:**
- Data: same 77 high-fidelity pairs as exp04
- Result shape per pair: (n\_sent, n\_layers, n\_heads) = (≤4, 32, 32)
- Elevation metric: mean(stego − open) averaged over pairs and K≥1, per (layer, head)

**Results:**

Top-5 heads by mean elevation across all layers:

| Head | Mean elevation | Peak layer |
|---|---|---|
| **28** | **0.0266** | **27** |
| 8  | 0.0169 | 11 |
| 27 | 0.0167 | 7  |
| 30 | 0.0163 | 11 |
| 29 | 0.0160 | 12 |

![Per-head elevation heatmap](exp04c/per_head_elevation.png)
![Best head curves](exp04c/best_head_curves.png)

**Conclusion:** The elevation is not diffuse — it is concentrated in a small number of heads organised in two clusters:

1. **Head 28, layer 27** (late layers): the strongest single contributor, score 60% above the next group. At this head, stego attention to acrostic positions reaches ≈0.5 while open remains near zero. Layer 27 is immediately upstream of the logit lens encoding onset (layer 29, exp02), suggesting this head may participate directly in letter commitment.

2. **Heads 8, 29, 30, layers 11–12** (mid layers): a cluster of four heads of approximately equal strength, consistent with the layers 10–15 signal observed in the exp04 heatmap. These heads likely track paragraph structure earlier in processing.

The remaining 27 heads show near-zero or negative elevation at their peak layers. The effect is concentrated in a small number of heads rather than being model-wide — however, see exp05a for a revision of this interpretation when averaged across all layers.

**Follow-up (H4):** causal verification via head ablation — zero out head 28 and the L11–12 cluster; test whether elevation drops and encoding fidelity degrades.

---

## exp05a - 2026-06-14

**Question:** How much of the exp04 elevation is attributable to head 28 and the L11–12 cluster? Is the signal disrupted by analytically removing the top-5 heads?

**Method:** Analytical ablation — same forward passes as exp04c; elevation recomputed under six head configurations without modifying the model.

**Setup:**
- Data: same 77 high-fidelity pairs as exp04
- Configurations: all\_heads, no\_head28, head28\_only, cluster2\_only, both\_clusters, background (27 heads excluding top-5)

**Results:**

| Configuration | n heads | Mean elev | % of baseline | t | p |
|---|---|---|---|---|---|
| all\_heads (baseline) | 32 | 0.01049 | 100% | 33.6 | <0.0001 |
| no\_head28 | 31 | 0.00997 | 95% | 32.6 | <0.0001 |
| head28\_only | 1 | 0.02659 | 254% | 44.8 | <0.0001 |
| cluster2\_only | 4 | 0.01648 | 157% | 38.3 | <0.0001 |
| both\_clusters | 5 | 0.01851 | 176% | 41.1 | <0.0001 |
| background | 27 | 0.00901 | 86% | 30.9 | <0.0001 |

![Head contribution](exp05a/head_contribution.png)

**Conclusion:** Head 28 has 2.5× higher per-head elevation than the model average (t = 44.8), confirming it as the strongest individual tracker. However, removing it reduces overall elevation by only 5%, and removing all top-5 heads leaves 86% of the baseline signal intact (t = 30.9, p < 0.0001). The elevation signal is a distributed property of the full attention mechanism, not a localised circuit.

This revises the exp04c interpretation: the heatmap showed peak elevation per layer, which highlighted top-5 heads. When averaged across all layers, the 27 background heads carry comparable aggregate signal. There is a gradient of intensity (top-5 heads are more specialised) but no sharp boundary between "tracking" and "non-tracking" heads.

**Implication for H4:** Zeroing the top-5 heads is insufficient to disrupt elevation. Causal verification requires either a broader intervention (many heads or full layers) or a generation fidelity test. From a detection standpoint, distributed signal is more robust: it cannot be evaded by suppressing a small number of heads.

---

## exp06_control - 2026-06-16

**Question:** Is the attention elevation from exp04–05a specific to steganographic hiding, or does it appear whenever the model is constrained to start sentences with specific letters?

**Method:** Generated a third condition — open acrostic instruction ("start each reasoning step with the letters S, A, F, E in order, one letter per step") — without the word "Hidden". Forward pass with `output_attentions=True`; elevation computed as (control − open) per pair. Compared against stego baseline from exp04/05a.

**Setup:**
- Data: same high-fidelity subset as exp04–05a (N_MAX=80); 70 pairs used after fidelity filtering
- n_skipped_gen: 9 (control fidelity < 0.75), n_skipped_attn: 1
- GPU: L4

**Results:**

| Condition | Elevation | % of stego baseline | t | p |
|---|---|---|---|---|
| open (baseline) | 0 | 0% | — | — |
| stego (exp04) | 0.010491 | 100% | 33.6 | <0.0001 |
| **control (exp06)** | **0.010054** | **95.8%** | **29.1** | **<0.0001** |

![Control vs stego elevation](exp06_control/control_vs_stego.png)

**Conclusion: falsifier triggered.**

Control elevation (95.8% of stego baseline) is statistically indistinguishable from stego elevation. The effect found in exp04–05a is driven by the structural constraint on sentence-start letters, not by the secrecy of the encoding instruction.

**Implication for exp04–05a:** the elevation signal is real and reproducible, but not diagnostic for steganography specifically. It measures "model follows a first-letter rule" regardless of whether that rule is hidden or explicit.

**Next steps:** H4 calibration (log-probabilities at acrostic positions) and H1 attention probe trained to distinguish stego vs control rather than stego vs open.

---

## exp07 - 2026-06-17

**Question (raised by colleague Tansylu):** does the model attend back to the prompt (where the letter constraint is specified), rather than between sentence starts within the CoT?

**Method:** two metrics measured in parallel on the same 70 pairs: (1) elevation as before; (2) `sum(attn[pos_K → 0:plen])` per head — fraction of attention allocated to prompt tokens.

**Results (n=70, skipped=1):**

| Metric | Stego | Control | Open | Stego vs ctrl |
|---|---|---|---|---|
| Elevation (rel. to open) | 0.01067 ± 0.000321 | 0.01009 ± 0.000338 | 0 | — |
| Prompt attention fraction | 0.856 ± 0.002 | 0.857 ± 0.002 | 0.836 ± 0.004 | t=−1.43, p=0.156 |

Prompt attention vs open: stego t=7.82 p≈0, control t=8.55 p≈0.

**Conclusion:** Constrained conditions (stego and control) allocate ~2 percentage points more attention to prompt tokens than open — directionally confirming the colleague's hypothesis. Stego ≈ control in both metrics. **Caveat:** stego/control prompts are ~30 tokens longer than open, which mechanistically confounds the comparison (a larger `plen` raises the sum independently of any specific attention to letter-instruction tokens). To isolate true instruction-seeking attention, one would need to compare attention to instruction tokens separately from neutral prompt tokens. This is a supporting observation for exp06_control, not a standalone result.

---

## exp08 - 2026-06-18

**Hypothesis (H4 — mentor, meeting June 15):** Log P of sentence-start tokens is anomalous in stego CoT relative to the open condition. The model is more confident about constrained letters.

**Method:** Teacher forcing — forward pass on already-generated text, no new generation. For each token i at position ≥ plen, log P is computed as `lp[i-1, token_ids[i]]`. Two comparisons:
1. Cross-condition: mean log P at sentence-start positions, stego vs open, paired t-test.
2. Within-sequence z-score: z-score of sentence-start log P relative to all other tokens in the same CoT — computed for both conditions to distinguish stego-specific from structural effects.

**Setup:**
- Data: `exp01/valid_pairs.json`, high-fidelity subset (fidelity = 1.0), n = 78 pairs (2 skipped: stego fidelity < 0.75 on regeneration)
- Open condition: taken from `pair['open_ids']` (exp01 data); stego condition regenerated to obtain log P of that specific generation
- GPU: L4

**Results:**

Metric 1 — cross-condition sentence-start log P:

| Condition | Mean log P | SE |
|---|---|---|
| open | −0.337 | 0.025 |
| stego | −0.061 | 0.007 |
| diff (stego − open) | +0.276 | 0.027 |
| t (paired) | 10.05 | p ≈ 0 |

Metric 2 — within-sequence z-score at sentence-start positions:

| Condition | Mean z | SE | t vs 0 | p |
|---|---|---|---|---|
| stego | +0.637 | 0.022 | 28.7 | ≈ 0 |
| open  | −0.470 | 0.087 | −5.4 | ≈ 0 |

![Log P calibration](exp08/exp08_logprob.png)

**Conclusion:** H4 confirmed. Sentence-start tokens are anomalously HIGH confidence within stego CoT (z = +0.637) and anomalously LOW confidence within open CoT (z = −0.470). The directions are opposite: in unconstrained reasoning, paragraph-initial tokens are points of high uncertainty; in stego, they are points of near-certainty (the letter is specified by the instruction). The effect is consistent across all 78 pairs. Requires a control condition to determine whether the effect is stego-specific or structural.

---

## exp08b - 2026-06-18

**Question:** Is the log P anomaly (exp08) specific to steganographic hiding, or does it appear whenever the model is given an explicit open letter constraint?

**Method:** Same teacher forcing as exp08; condition replaced with open acrostic instruction ("start each reasoning step with the letters S, A, F, E in order, one letter per step"), without the word "Hidden". Compared against exp08 stego (z = +0.637) and open (z = −0.470) reference values.

**Setup:**
- n = 71 pairs (9 skipped: control fidelity < 0.75)
- exp08 reference z-values loaded from `exp08/exp08_summary.json` or used as hardcoded fallback
- GPU: L4

**Results:**

Metric 1 — cross-condition sentence-start log P:

| Condition | Mean log P | diff vs open | t | p |
|---|---|---|---|---|
| open    | −0.345 | —      | —    | —   |
| control | −0.092 | +0.253 | 8.99 | ≈ 0 |
| stego (exp08) | −0.061 | +0.276 | 10.05 | ≈ 0 |

Metric 2 — within-sequence z-score (all three conditions):

| Condition | Mean z | Source |
|---|---|---|
| open (baseline) | −0.470 | exp08 |
| **control**     | **+0.514** | exp08b |
| stego           | +0.637 | exp08 |

![Log P control](exp08b/exp08b_logprob_control.png)

**Conclusion: falsifier triggered.**

Control z (+0.514) is nearly equal to stego z (+0.637); both are far from open (−0.470) and in the same direction. The log P anomaly is a structural property of any letter constraint, not of steganographic hiding — the same finding as exp06_control for attention.

**Overall conclusion for exp04–exp08b:** both signal types studied — attention elevation and log P elevation at sentence-start positions — are structural properties of first-letter constraints. They measure "model is following a letter rule" regardless of whether that rule is hidden or explicit. Neither signal distinguishes stego from control conditions.

---

## exp10 - 2026-06-20

**Question:** Do attention entropy and within-KL divergence at sentence-start positions show a signal for explicit X- steganography (exp01 data) on Llama-3.1-8B-Instruct?

**Method:** Teacher forcing on 80 stego/open pairs (exp01, fidelity=1.0). Metrics: (1) Shannon entropy of attention weights at sentence-start positions; (2) within-KL = KL(attention at sentence-starts || mean attention at other positions in same sequence).

**Setup:** n=80 pairs, Llama-3.1-8B-Instruct, bfloat16, A100, `attn_implementation='eager'`.

**Results:**

| Metric | Stego | Open | Diff | t | p |
|---|---|---|---|---|---|
| Entropy | 3.057 | 2.910 | +0.147 | 12.1 | ≈0 |
| Within-KL | 0.412 | 0.209 | +0.203 | 29.8 | ≈0 |

Top-5 heads by within-KL diff: L8H11, L10H2, L14H17, L27H28, L29H9.

**Conclusion:** Both signals are significant. Whether they are stego-specific or structural is addressed by exp10b.

---

## exp10b - 2026-06-20

**Question:** Is the attention signal from exp10 stego-specific or structural?

**Method:** Same metrics as exp10, control condition (explicit X- but visible, no "Hidden") vs open. n=40 usable pairs.

**Results:**

| Metric | Control | Open | Diff | t | p |
|---|---|---|---|---|---|
| Entropy | 3.029 | 2.907 | +0.122 | 10.2 | ≈0 |
| Within-KL | 0.400 | 0.193 | +0.207 | 29.1 | ≈0 |

Same top heads as exp10 (L8H11, L27H28, L29H9, L10H2). Control ≈ stego on both metrics.

**Conclusion:** Signal is structural — driven by the letter constraint, not by its hiddenness. Replicates exp06_control finding on Llama-3.1-8B with new metrics.

---

## exp11 - 2026-06-20

**Dataset:** Natural hidden acrostic CoT pairs for Llama-3.1-8B-Instruct mechanistic analysis.

**Motivation:** exp01 data used explicit `X - ` markers (trivially regex-detectable). exp11 uses natural format: required letter is simply the first letter of the first word of each paragraph, with no visual marker.

**Generation:** ARC-Challenge tasks, 8 four-letter payloads (SAFE/LINK/CODE/OPEN/LOCK/PLAN/HIDE/MARK), Llama-3.1-8B-Instruct, greedy decoding. Both stego and open conditions generated by same model.

**Filter:** fidelity=1.0 (all 4 letters correct), exactly 4 paragraphs (double-newline separated), 0 explicit markers.

**Result:** 659 pairs, all fidelity=1.0, 0 explicit markers.

**Files:** `exp11/exp11_pairs.json`, `exp11/DATASET_CARD.md`

Pilot (exp11b, 20 pairs): 18/20 valid, mean fidelity=0.99 → full generation confirmed viable.

---

## exp12 - 2026-06-20

**Question:** Do attention signals (entropy + within-KL) persist with natural hidden acrostics (no X- markers)?

**Method:** Same as exp10 but on exp11 data (paragraph-start positions instead of sentence-start). Stego vs open only — control condition missing (design flaw corrected in exp13b).

**Setup:** n=641 pairs (659 − 18 skipped), Llama-3.1-8B-Instruct, A100.

**Results:**

| Metric | Stego | Open | Diff | t | p | vs exp10 |
|---|---|---|---|---|---|---|
| Entropy | 3.431 | 3.190 | +0.241 | 54.1 | ≈0 | ↑ grew |
| Within-KL | 0.346 | 0.304 | +0.042 | 15.9 | ≈0 | ↓ shrank |

**Conclusion:** Entropy signal persists (grew, partially confounded by longer sequences). Within-KL gap shrank substantially (0.203→0.042): (1) no X- marker → paragraph starts less salient; (2) selection bias — requiring 4 paragraph starts in open filtered for naturally structured open texts. No control condition → stego-specificity untested.

**Files:** `exp12/exp12_summary.json`, `exp12/exp12_per_layer_bars.png`

---

## exp13b - 2026-06-21

**Question:** Are attention signals (entropy + within-KL at paragraph-start positions) stego-specific or structural?

**Method:** Teacher forcing with separate forward pass: `output_attentions=True, output_hidden_states=False`, `attn_implementation='eager'`. Three conditions: stego/open/control. Stego and control validated to exactly n_para paragraphs; open accepted with any count ≥ 1 (unconstrained reasoning).

**Setup:** n=622 triplets, 0 skipped, Llama-3.1-8B-Instruct, A100.

**Results:**

| Metric | Stego | Open | Control | S−O diff | S−C diff | p(S vs C) |
|---|---|---|---|---|---|---|
| Entropy (bits) | 3.433 | 3.314 | 3.427 | +0.119 | +0.006 | 0.037 |
| Within-KL (nats) | 0.564 | 0.593 | 0.567 | −0.029 | −0.003 | 0.281 |

S−C diffs are 5% (entropy) and 10% (within-KL) of the stego−open gap. Within-KL is not significant (p=0.281). Entropy difference (p=0.037) is marginal and tiny in effect size.

Note: open has higher within-KL than stego/control because open responses are unconstrained and paragraph-start positions occur in more diverse sequential contexts.

**Key conclusion:** stego ≈ control. Attention signals measure "letter constraint is present", not "letter constraint is hidden". Neither metric can distinguish steganographic from visible constraints.

**Files:** `exp13b/exp13b_summary.json`, `exp13b/exp13b_entropy.png`, `exp13b/exp13b_within_kl.png`

---

## exp13c - 2026-06-21

**Question:** At which layer does the model commit to the constrained letter? Is the layer-by-layer P(letter) trajectory the same for stego and control?

**Method:** Teacher forcing with separate forward pass: `output_attentions=False, output_hidden_states=True`. At each of 32 layers, hidden state h[l][pos−1] projected through `lm_head(RMSNorm(h))` → P(letter) summed over all vocab tokens whose stripped decoded form starts with the required letter. Open condition trimmed to first n_para paragraph starts to align with payload positions.

**Setup:** n=604 triplets, 18 skipped, Llama-3.1-8B-Instruct, A100.

**Results (mean P(letter) across all layers):**

| Condition | Mean P(letter) |
|---|---|
| stego | 0.1017 |
| control | 0.1026 |
| open | 0.0361 |

stego vs control: diff=−0.00092, t=−4.2, p=2.9e-5 (significant but 1.4% of stego−open gap).

**Layer profile:**

| Layer | stego | control | open |
|---|---|---|---|
| 0–15 | ~0.032 | ~0.032 | ~0.030 |
| 16 | 0.047 | 0.048 | 0.030 |
| 18 | 0.100 | 0.103 | 0.033 |
| 22 | 0.144 | 0.142 | 0.029 |
| 30 | 0.303 | 0.315 | 0.023 |
| **31** | **0.968** | **0.968** | **0.011** |

**Key conclusions:**
1. Commitment builds gradually from L16, crystallises sharply at L30–L31 (P = 0.968 for both stego and control, vs 0.011 for open).
2. Stego and control follow an identical computational trajectory across all 32 layers. The model "decides" the constrained letter in exactly the same way whether the constraint is hidden or visible.
3. A logit-lens-based loss function cannot distinguish steganographic encoding from a visible letter constraint.

**Answer to weekly question (mentor, June 15):** Vectors do "sway towards the constrained letter" starting at L16, but equally for hidden and visible constraints. The swaying is structural, not stego-specific. Logit lens cannot serve as a steganography detector.

**Files:** `exp13c/exp13c_summary.json`, `exp13c/exp13c_logit_lens.png`, `exp13c/exp13c_stego_minus_control.png`

---

## exp09c - 2026-06-22

**Question:** When a word mid-sequence is replaced (changing the starting letter by ≥ 4 alphabet positions), does GPT-4o continue the chain from the *patched* letter (text-following) or from the *originally planned* letter (plan-following)?

**Design:** Thought-Anchors-style text-level patching applied mid-sequence in an alphabet+1 chain. A valid 6-sentence chain is generated; sentence K (patch position) is modified by replacing its first word. Three trial types per chain × position:
- **FORCED** — first word replaced with a word starting with a wrong letter (≥ 4 positions away in the alphabet)
- **IDENTITY** — sentence K re-fed unchanged (control: editing artefact alone does not perturb)
- **WORD\_SWAP** — first word replaced with a different word starting with the *same* correct letter (control: wording change without letter change)

Outcome for K+1 is labelled **text** (follows patched letter's successor), **plan** (follows original planned letter), or **neither** (chain abandoned or wrong).

**Setup:**
- Model: GPT-4o (no thinking mode; temperature 0 for continuation)
- Rule: alphabet+1 (each sentence starts with the next alphabet letter)
- N chains: 20 (start letter drawn from 16 "safe" letters; JKQWXZ excluded)
- Patch positions K: 2, 3, 4 (sweeping mid-chain positions)
- Total trials per type: 60 (20 chains × 3 positions)
- Cascade: outcome also recorded for position K+2

**Results:**

| Trial type | text | plan | neither | p (binomial, text > 0.5) |
|---|---|---|---|---|
| FORCED     | 49/60 (81.7%) | 1/60 (1.7%) | 10/60 (16.7%) | < 0.000001 ✓ |
| IDENTITY   | 60/60 (100%)  | 0/60        | 0/60          | — |
| WORD\_SWAP | 60/60 (100%)  | 0/60        | 0/60          | — |

FORCED by patch position (K+1):

| K | text | n |
|---|---|---|
| 2 | 17 (85%) | 20 |
| 3 | 16 (80%) | 20 |
| 4 | 16 (80%) | 20 |

Cascade K+2 (FORCED): text = 49/60 (81.7%), plan = 1/60 — the text-following behaviour propagates one step further with the same rate.

**Conclusion:** Mid-sequence word replacement causally determines the next letter in the chain. GPT-4o reads its own generated text and applies the alphabet+1 rule to the patched letter rather than executing a pre-planned sequence. The IDENTITY and WORD\_SWAP controls show near-zero perturbation (100% text continuation), confirming that the effect is driven by the letter change, not by editing artefacts. The 16.7% "neither" rate in FORCED conditions reflects occasional chain abandonment (the model breaks task framing), not plan-following. The cascade result (K+2 also follows the patched branch at 81.7%) indicates that the error propagates forward — once the patched letter is accepted, subsequent steps continue from it.

This extends exp09b (single-sentence patch on S1) to mid-chain positions (K = 2, 3, 4), confirming the causal role of generated text across the full chain length.

**Files:** `exp09c/exp09c_results.json`

---

## exp09d - 2026-06-22

**Question:** Can the text-following effect (exp09b/exp09c) be quantified as a full conditional letter distribution, and is it distinguishable from chance (placebo) via KL divergence?

**Design:** Prefill-based distributional patching. For each trace:
1. A valid 6-sentence alphabet+1 chain is generated.
2. Sentence S2 is replaced with a forced (wrong-letter) or placebo (same-letter, different-word) variant.
3. The model is asked to continue from S3 onward, with S1 + S2\_variant as a prefill in the assistant turn.
4. S3 is sampled N\_SAMPLES times (temperature 0.9) per condition.
5. KL divergence from the forced/placebo letter distribution to the base (unmodified) distribution is computed.

Three conditions per trace: **base** (S2 unchanged), **force** (S2 first word → wrong letter ≥ 4 away), **placebo** (S2 first word → different word, same letter).

Read-back rate = fraction of FORCE samples where S3 starts with next\_letter(force\_letter) rather than next\_letter(correct\_letter).

**Setup:**
- Model: GPT-4o (temperature 0.9 for sampling)
- Rule: alphabet+1; 16 safe letters; force letter ≥ 4 positions away
- N traces: 20; N samples per trace: 20; total FORCE samples: 400
- KL smoothing: Laplace (ε = 0.5) over all 26 letters
- Seed: 42

**Results:**

| Metric | Value |
|---|---|
| Valid traces | 20 / 20 |
| Read-back rate (FORCE, mean ± std) | 81.0% ± 28.7% (min 0%, max 100%) |
| KL force vs base (mean ± std)   | 2.1099 ± 0.1412 |
| KL placebo vs base (mean ± std) | 0.0010 ± 0.0044 |

Across all 400 FORCE samples:

| Outcome | Count | % |
|---|---|---|
| Read-back (follows patched letter) | 324 | 81.0% |
| Plan (follows original letter)     | 1   | 0.2% |
| Neither                            | 75  | 18.8% |

Binomial test (read-back > plan, n\_clear = 325): p < 0.000001.

KL\_force / KL\_placebo ratio: 2110 × — placebo is negligibly different from base; forced patching shifts the letter distribution by more than 2 nats.

**Anomalous trace (start=D, 0% read-back):** force letter was R → expected S3 starting with S, but the model produced E (= next\_letter(D)) in all 20 samples. The model appears to re-anchor to the chain start letter (D) and resume the baseline sequence, bypassing both the force and the plan. This is a systematic prompt-interpretation artefact (model counts from the start rather than reading S2), not random noise. One out of 20 traces; does not affect the aggregate result.

**Conclusion:** The letter distribution of S3 shifts dramatically when S2's first letter is changed (KL = 2.11) and is unchanged when only the wording changes (KL = 0.001). This 2100× ratio constitutes strong quantitative evidence that the first letter of each generated sentence is causally determined by the first letter of the preceding sentence, consistent with text-following (read-back) rather than plan-following (prompt-only reasoning).

Together with exp09b (100% text-following, patching S1) and exp09c (81.7%, mid-chain patching), exp09d confirms that acrostic encoding positions are causally load-bearing across three complementary methodologies: discrete outcome labelling, mid-chain position sweep, and full distributional KL measurement.

**Files:** `exp09d/exp09d_results.json`, `exp09d/figures/`

---

## exp13d - 2026-06-26

**Question (mentor, meeting 2026-06-15):** Does a paragraph-start token K attend more to (a) previous paragraph-start tokens {1..K-1} and (b) prompt tokens that specify the letter scheme, in stego vs open vs control conditions?

**Motivation:** The mentor's narrowed attention hypothesis: if the model is doing constrained letter generation, it must consult (a) the preceding constrained letters to maintain the sequence and (b) the prompt scheme to know the target letter. If stego > baseline on either metric, this implicates these token sets in the encoding mechanism. If stego ≈ control, the effect is structural (driven by any letter constraint, not by its hiddenness).

This experiment refines exp04–exp06: instead of averaging over all heads and measuring elevation relative to mid-sentence positions, it measures directed attention from paragraph-start token K to specific source sets, on the natural acrostic dataset (exp11, no X- markers).

**Setup:**
- Data: `exp11/exp11_pairs_with_control.json`, 604 triplets (stego / open / control), 0 skipped
- Model: Llama-3.1-8B-Instruct, `output_attentions=True`, `attn_implementation='eager'`
- Source set 1 (prev\_starts): positions of previous paragraph-start tokens {sent\_positions[0..K-1]}
- Source set 2 (scheme): prompt token span encoding the letter constraint (payload + instruction)
- Metric: mean attention weight from position K to source set, averaged over all heads and pairs

**Results:**

Attention to previous paragraph-start tokens:

| Comparison | Mean diff | t | p |
|---|---|---|---|
| stego vs open    | −0.00395 | −22.24 | ≈ 0 |
| stego vs control | +0.00023 | +2.23  | 0.026 |
| control vs open  | −0.00418 | −24.30 | ≈ 0 |

Attention to scheme tokens:

| Comparison | Mean diff | t | p |
|---|---|---|---|
| stego vs control | −0.00066 | −12.16 | ≈ 0 |

Peak attention weight per layer (prev\_starts):

| Condition | Peak layer | Peak value |
|---|---|---|
| stego   | 15 | 0.0677 |
| control | 12 | 0.0664 |
| open    | 16 | 0.0574 |

Peak attention to scheme tokens: layer 14 for both stego (0.1472) and control (0.1438).

**Interpretation of direction:**

The stego < open result for prev\_starts (diff = −0.004) is counter-intuitive but coherent: unconstrained (open) tokens attend more to previous paragraph-start positions because they must maintain topic continuity without a scheme — paragraph-start tokens are their primary contextual anchors. Constrained (stego, control) tokens already know the target letter from the prompt scheme, so they rely less on reading back previous constrained positions.

Both the "prev\_starts" and "scheme" metrics show stego ≈ control: the stego–control differences are statistically detectable (t = 2.23 / −12.16) but effects are trivially small relative to the stego–open gap (stego–control diff is 5.8% / 16.7% of the stego–open gap for the respective metrics).

**Conclusion:** The directed attention pattern is a structural property of letter-constrained generation. Neither the attention back to previous paragraph-start positions nor the attention to the prompt's letter-scheme tokens can distinguish steganographic from openly-constrained letter generation. This replicates the finding of exp06 (attention elevation), exp08b (log P), exp10b (entropy/within-KL), and exp13b/c (entropy, within-KL, logit lens) on a fourth independent metric and the natural acrostic dataset.

**Overall mechanistic conclusion (exp04–exp13d):** Across six independent mechanistic signals — attention elevation, head specialisation, log P z-score, attention entropy, within-KL divergence, logit-lens P(letter), and directed attention from constrained positions — no signal distinguishes a hidden letter constraint from a visible one. All signals measure "the model is following a letter rule," not "the model is hiding a message."

**Files:** `exp13d/exp13d_summary.json`, `exp13d/exp13d_raw.json`, `exp13d/exp13d_prev_starts.png`, `exp13d/exp13d_scheme.png`

---

## exp14 - 2026-06-29

**Question:** Which token positions are causally necessary for the model to predict the correct paragraph-start letter? Specifically: does the model rely on (a) the immediately preceding token, (b) previous paragraph-start positions, or (c) the letter tokens in the prompt scheme?

**Motivation:** exp13d measured directed attention observationally. exp14 answers the same question causally: zero out attention to each source set and measure the drop in `P(letter)` at the paragraph-start prediction position. The mentor's hypothesis (meeting 2026-06-22): "the acrostic token must attend to the previous token and to the scheme tokens." exp14 directly tests this.

**Method:** Monkey-patching `eager_attention_forward` (transformers 5.9.0) on Llama-3.1-8B-Instruct (A100, bfloat16). Teacher-forced on `exp11/exp11_pairs_with_control.json`, 120 pairs × 2 conditions = 240 cases. Metric: `P(letter)` = sum of softmax probability over all tokens whose decoded form begins with the required letter, at the paragraph-start prediction position. Paired statistics: Cohen dz = mean(delta_i)/std(delta_i).

**Five sub-experiments (120 pairs each):**

### exp14a — Source masking

| Source masked | stego delta | control delta | dz (stego) |
|---|---|---|---|
| `prev_starts` (previous paragraph-start positions) | +0.188 | +0.167 | 1.14 |
| `letters_only` (payload letter tokens in scheme) | +0.371 | +0.411 | 3.37 |
| `full_scheme` (entire scheme span) | +0.771 | +0.809 | 11.52 |

Baseline P ≈ 0.96. After full_scheme masking: P drops to 0.15–0.19. After letters_only: to 0.55–0.59.

### exp14b — Previous token vs long-range

| Source masked | stego delta | control delta | dz (control) | p (control) |
|---|---|---|---|---|
| `prev_token_only` (query_pos − 1) | +0.016 | +0.003 | 0.14 | **0.141 — not significant** |
| `prev_starts_only` | +0.188 | +0.167 | 1.20 | ≈0 |
| `both` | +0.204 | +0.173 | 1.10 | ≈0 |

Key result: the immediately preceding token is causally irrelevant for the control condition (p = 0.141). Adding it on top of `prev_starts` changes nothing (`both` ≈ `prev_starts_only`).

### exp14c — Scheme granularity

| Source masked | stego delta | control delta | dz (stego) |
|---|---|---|---|
| `letters_only` | +0.371 | +0.411 | 3.37 |
| `scheme_without_letters` | +0.030 | +0.033 | 1.29 |
| `full_scheme` | +0.771 | +0.809 | 11.52 |

The specific letter tokens (S, A, F, E) carry ~90% of the scheme's causal signal.

### exp14d — Head specialization

Top-8 heads (ranked on first 60 pairs): L16H1 (0.755), L14H18 (0.550), L18H16 (0.533), L19H3 (0.488), L15H12 (0.487), L14H29 (0.439), L14H22 (0.433), L15H16 (0.408).

| Ablation | stego delta | control delta | dz (stego) |
|---|---|---|---|
| Top-8 heads | +0.068 | +0.057 | 1.07 |
| Random 8 heads (mean over 8 sets) | +0.003 | +0.002 | — |

Top-8 heads are ~20–25× more causal than random heads. Absolute effect modest (7pp) compared to full-scheme masking (81pp).

### exp14e — Attention rewrite

| Source rewritten | stego delta | control delta | dz (stego) |
|---|---|---|---|
| `rewrite_letters_uniform` | +0.615 | +0.657 | 6.15 |
| `rewrite_full_scheme_uniform` | +0.819 | +0.850 | 11.19 |
| `rewrite_prev_starts_uniform` | +0.217 | +0.208 | 1.68 |

Rewriting (redirecting mass) consistently stronger than masking (zeroing); same source hierarchy.

**Information source hierarchy:**
```
full_scheme (dz ≈ 12) ≫ letters_only (dz ≈ 3.4) > prev_starts (dz ≈ 1.2) ≫ prev_token (dz ≈ 0.1, n.s.)
```

**Key conclusions:**
1. The model reads the letter **directly from the explicit letter tokens in the prompt scheme** — the primary causal source (dz ≈ 12).
2. Previous paragraph-start positions are a secondary causal source (dz ≈ 1.2, ~17–19pp drop).
3. The immediately preceding token is **causally irrelevant** (p = 0.141 for control) — refuting the mentor's "previous token" part of the hypothesis.
4. Head specialisation exists but the top-8 heads carry only ~7pp of causal weight vs 81pp for the full scheme — they participate in routing but are not the primary mechanism.
5. **stego ≈ control across all 12 conditions** (stego/control delta ratio 0.90–1.18). The causal mechanism is identical whether the constraint is hidden or visible.

This is the seventh independent signal (after exp04–exp13d) confirming the structural, non-steganographic character of the acrostic encoding mechanism. Causal source masking provides the strongest possible mechanistic evidence: removing the scheme tokens destroys letter prediction nearly completely (P drops from 0.96 to 0.15).

**Files:** `exp14/exp14_summary.json`, `exp14/exp14a_source_masking_summary.json`, `exp14/exp14b_prev_token_vs_long_range_summary.json`, `exp14/exp14c_scheme_granularity_summary.json`, `exp14/exp14d_head_specialization_summary.json`, `exp14/exp14e_attention_rewrite_summary.json`, `exp14/exp14{a–e}_*.png`

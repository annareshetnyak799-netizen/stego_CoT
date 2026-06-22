# Steganographic read-back — results analysis

**Run:** `readback_dist/20260622_164505` · **Model:** Claude Sonnet 4.5 · **Date:** 2026-06-22
**Design:** counterfactual importance (Thought-Anchors-style resampling) re-targeted to a hidden acrostic payload.

---

## Hypothesis
**Steganographic read-back is causally load-bearing.** When the model hides a payload in its sentence-initial
letters, each sentence's first letter is *derived from the letter it actually emitted in the previous
sentence* (it reads its own output) — not from a plan fixed at generation start.

Operational form: forcing one sentence's encoded letter to a *wrong* value should shift the **distribution**
of the next sentence's first letter; a cosmetic edit that *preserves* the letter should not.

## Scope
- **One model:** Sonnet 4.5 — the frontier model that supports both temperature *and* prefill. (Opus 4.8 /
  Sonnet 4.6 reject both, so they cannot produce the sampled distributions this method requires.)
- **One channel:** acrostic (first letter of each sentence), number→letter key, +1 chain, K=4 (5 sentences).
- **One intervention point:** force sentence 2, read sentence 3; 30 everyday cover questions.
- A **read-back / mechanism** claim — *not* a claim that the model can covertly *compute* the chain
  (established separately: it copies / looks up the key symbol, it does not do hidden arithmetic).

## Approach
Borrow **counterfactual importance** from *Thought Anchors* (Bogdan et al., arXiv:2506.19143): resample a
unit, measure the shift in the output distribution via KL. Two adaptations:
1. **Re-target the "output"** from the task answer to the **hidden payload** (the next encoded letter).
2. Replace free resampling with a **decomposed, placebo-controlled** counterfactual that isolates the
   encoded *symbol* as the causal variable.

## Specific methodology
1. **Base traces:** generate correct acrostic answers; keep **35** that are exactly correct (~60% base success).
2. **Three conditions per trace**, each **sampled S=40 times at temperature** (to estimate a *distribution*,
   not a single point — this is why temperature, hence Sonnet 4.5, is required):
   - **BASE** — original sentence 2, resampled → P0 (natural next-letter variation).
   - **FORCE** — sentence 2's first letter set to a *wrong* letter → P_force.
   - **PLACEBO** — sentence 2 reworded, *same* first letter → P_placebo.
3. **Metric:** `KL(P_force‖P0)` vs `KL(P_placebo‖P0)`, add-α smoothed, computed per trace.
4. **Inference:** bootstrap 95% CIs over **traces** (the cluster unit; the 40 samples within a trace are not
   independent); **permutation test** on `Δ = KL_force − KL_placebo` (5,000 sign-flips).
5. **Secondary readouts:** *read-back mass* = share of forced samples landing exactly on the read-back-
   predicted letter; per-trace outcome = read-back / original / off-chain.
6. Full provenance saved: every prompt, prefill, and sampled continuation (`base_traces.jsonl`,
   `dists.jsonl`, `raw_samples.jsonl`).

## Definitions (what "KL" means here)

For one trace we look at a single random variable: the **first letter of the next sentence** (sentence 3).
It takes values in the alphabet $\mathcal{X} = \{\texttt{S,C,P,A,B,T,D,M,R,F},\,\_\}$ (the 10 code letters
plus `_` = "no sentence produced"). Each condition gives an **empirical distribution** over $\mathcal{X}$,
estimated from $S=40$ sampled continuations and add-$\alpha$ smoothed ($\alpha=0.5$) so no probability is 0:

$$P(x) \;=\; \frac{\text{count}(x) + \alpha}{S + \alpha\,|\mathcal{X}|}.$$

- $P_0$ — **base** (sentence 2 untouched): the natural distribution of the next letter.
- $P_{\text{force}}$ — sentence 2's first letter forced to a **wrong** value.
- $P_{\text{plac}}$ — sentence 2 reworded, **same** first letter.

**Kullback–Leibler (KL) divergence** measures how far one distribution is from another (in nats; $0$ iff
identical, larger = more different):

$$D_{\mathrm{KL}}(P \,\|\, Q) \;=\; \sum_{x \in \mathcal{X}} P(x)\,\log\frac{P(x)}{Q(x)}.$$

So our two quantities, per trace, are how far each *edit* pushed the next-letter distribution away from the
untouched one:

$$\text{shift}_{\text{force}} = D_{\mathrm{KL}}(P_{\text{force}} \,\|\, P_0),\qquad
  \text{shift}_{\text{plac}}  = D_{\mathrm{KL}}(P_{\text{plac}} \,\|\, P_0).$$

The **read-back effect** removes the cosmetic baseline by taking the difference, then we test it across traces:

$$\Delta_i \;=\; D_{\mathrm{KL}}(P_{\text{force}}^{(i)} \,\|\, P_0^{(i)}) \;-\; D_{\mathrm{KL}}(P_{\text{plac}}^{(i)} \,\|\, P_0^{(i)}),
  \qquad \bar{\Delta} = \tfrac{1}{n}\sum_{i=1}^{n}\Delta_i .$$

**Permutation test** (is $\bar{\Delta}>0$ beyond chance?): randomly swap the force/placebo labels within each
trace, recompute the mean difference 5,000 times to build a null distribution, and report
$p = \Pr_{\text{null}}(\text{mean} \ge \bar{\Delta})$.

**Read-back mass** (secondary, interpretable): the probability the forced distribution puts on the *exact*
predicted read-back letter $r$, where $r$ is the letter for $(\text{number of the forced letter} + 1) \bmod 10$:

$$\text{read-back mass} = P_{\text{force}}(r).$$

Intuition: $D_{\mathrm{KL}}(P_{\text{force}}\|P_0)$ large + $D_{\mathrm{KL}}(P_{\text{plac}}\|P_0)\approx 0$
means **changing the letter** moves the next letter a lot while **changing only the wording** does not — i.e.
the model is reading the encoded letter.

## Results

| measure | mean [95% CI] |
|---|---|
| KL(force ‖ base)    | **2.78 [2.41, 3.12]** |
| KL(placebo ‖ base)  | **0.21 [0.05, 0.45]** |
| Δ (force − placebo) | **2.57 [2.13, 2.99]** |
| permutation p       | **0.0002** (test floor) |
| read-back mass      | **60% [48%, 72%]** |

**The core question:** when we tamper with sentence 2, does the model **follow its memorized plan** (emit the
letter it would have anyway) or **dynamically adjust to the new sentence** (recompute from what's actually
there)? Read-back = dynamic adjustment.

**Where the next letter lands (avg over traces):**
| condition | followed memorized plan (original letter) | adjusted to new sentence (read-back letter) |
|---|---|---|
| untouched (base) | **96%** | ~0% |
| placebo (same letter, reworded) | **92%** | ~0% |
| forced (wrong letter) | 10% | **60%** |

→ With nothing changed, or only the *wording* changed, the model follows the plan (96% / 92%). The moment the
*letter* is changed, it **dynamically adjusts** to the new (wrong) letter (60%) and abandons the memorized
plan (down to 10%). That is the read-back effect in one table.

**Per-trace structure (n=35):**
- Force **moves** the distribution in **32/35 (91%)**; placebo leaves it **unchanged** in **31/35 (89%)** →
  clean double dissociation.
- Dominant forced outcome: **read-back 22/35 (63%)**, original (precomputed) **4/35 (11%)**, off-chain **9/35 (26%)**.
- Read-back mass: **median 72%**, range 0–100% (strong but heterogeneous).

![distribution shift](figure.png)

*Figure — left:* where the next letter lands per condition (orange = original/correct-chain letter, green =
read-back letter); untouched & placebo stay orange, forced jumps green. *Middle:* KL distance of each edit's
next-letter distribution from untouched (forced ≈ 2.78 vs placebo ≈ 0.21; mean ±95% CI). *Right:* the
force−placebo gap (red) against the permutation null — far outside chance, $p=0.0002$.

## Interpretation
**Read-back is detected, decisively** (Δ = 2.57, p at the permutation floor): forcing the encoded letter
moves the next-letter distribution ~13× more than a same-letter reword. The placebo rules out "any edit
perturbs it" — the **encoded symbol itself** is the cause. The channel is a **live, load-bearing register**:
the model reads the letter it just wrote and continues from it, rather than following a fixed plan.

The ~40% that is not clean read-back is two *understood* failure modes, not noise:
- **off-chain (26%)** — the model drops the acrostic and writes natural prose;
- **original (11%)** — it keeps the precomputed letter; traced to a confound where the cover topic
  over-determines the next opener *and that letter equals the original* (e.g. "should children learn to
  code?" → sentence 3 wants "Coding…" = C = original), so forcing cannot dislodge it. Compounded by a crude
  first-word-swap that sometimes produced ungrammatical forced sentences the model ignored.

## Limitations
- **Single model, single position, K=4, one question set** — not yet a generalization claim.
- **Forcing used a crude first-word swap**, which occasionally produced incoherent forced sentences (e.g.
  "Recently who learn coding develop…") that the model ignored → inflates off-chain/original misses. So
  **60% read-back mass is a lower bound**. A coherent-rephrase forcing (rewrite the sentence with a cheap
  model so it naturally starts with the target letter) is implemented and verified; the cleaned re-run was
  not executed.
- **N = 35 traces** is the inferential unit — well-powered for *detection* (p at floor); rate CIs ≈ ±12–16%.
- Demonstrates **read-back**, not hidden computation.

## Next steps
1. Cleaned re-run with **coherent-rephrase forcing** (reuse base samples, re-sample force+placebo only ≈ $8).
2. **Position sweep** (force sentences 1–4) — does read-back hold across positions / cascade downstream?
3. **Mechanistic** (open model, e.g. Llama-3.1-8B): attention-suppression at the encoding sentence — the
   causal double-dissociation that would localize the read-back circuit (Thought Anchors Method 3).

## Artifacts (this folder)
`base_traces.jsonl` · `dists.jsonl` · `raw_samples.jsonl` (4,200 generations) · `stats.json` · `figure.png` ·
`report.md` · `viewer.html` (browsable, grouped by prompt → condition → outcome).

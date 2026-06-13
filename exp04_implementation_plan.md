# exp04 Implementation Plan — Stego-reasoning detection on Thought-Anchors

## The one finding that shapes everything
The repo has **two different intervention mechanisms**, and they are NOT interchangeable:
- **Attention-input suppression** (`hooks.py:270 apply_qwen_attn_mask_hooks`): masks attention *to* a
  token range, in a **forward pass over a fixed transcript** (teacher-forced, no generation). ← the right tool.
- **Head-output ablation** (`ablation.py:16 AttentionHeadAblator`): zeroes whole head outputs, supports
  `model.generate()`. Different thing; not what we want.

⇒ Our causal test = **teacher-forced next-token read on the fixed stego transcript, with vs without the mask** — not a regeneration. Read the predicted acrostic letter at `boundaries[i][0]-1` (token before each sentence-initial letter) and measure the accuracy drop. This is cheap and is the headline experiment.

## Reuse as-is (file : function)
- **Model:** `model_loader.py:167 get_deepseek_r1("qwen-14b", float32=True)` → (model, tok). Forces `eager` attn (needed for `output_attentions`).
- **Capture attention:** `analysis.py:157 analyze_text(...)` (one forward pass → attention_weights, tokens, logits); `attn_funcs.py:468 get_attention_to_step(...)` (attention *to* a chosen sentence = ~analysis A); `attn_funcs.py:336 get_avg_attention_matrix(...)`.
- **Suppression:** `hooks.py:270 apply_qwen_attn_mask_hooks(model, token_range, layer_2_heads_suppress)` driven by `analyze_text(token_range_to_mask=..., layers_to_mask=...)`; logits via `logits_funcs.py:188 analyze_text_get_p_logits`.
- **Receiver heads:** `receiver_head_funcs.py:72 get_all_heads_vert_scores` + `:65 get_3d_ar_kurtosis` + `:105 get_top_k_layer_head_kurts`.

## Data contract (the bridge)
Avoid the brittle on-disk rollout layout (`chunks.json` + `sanity_check_sentences`). Use the low-level path: everything we need takes **`(text, sentences)` in memory**.
- `attn_funcs.py:103 get_sentence_token_boundaries(text, sentences, "qwen-14b")` → token spans.
- We generate summaries as **one sentence per chain step**, so segmentation is exact and index-aligned.
- Encoding token for sentence i = `boundaries[i][0]`. Suppress with `token_range_to_mask=[[b,b+1], ...]`.
- Gotcha: each `sentences[i]` must be a verbatim slice of `text` or the boundary builder raises.

## Write-new (small)
1. `tasks.py` — modular chain `s_i=(s_{i-1}·a_i+b_i) mod 10`; random digit→letter code; prompt builders for Plain / Stego / Cover-only.
2. `generate.py` — thin `model.generate` wrapper; emit `{text, sentences, encoding_idxs, true_digits, code, answer}`.
3. `score.py` — hidden-task accuracy (decode acrostic), encoding fidelity, covertness monitor (LLM over visible summary only).
4. `suppression_task_effect.py` — fork of `attn_supp_funcs.py:28`, swap KL→hidden-task accuracy (drop) vs cover-task survival (KL stays low). Reuse `logits_funcs.py:11 decompress_logits_for_position` + `attn_supp_funcs.py:132 calculate_kl_divergence_sparse`.
5. Receiver-head aggregation over our in-memory transcripts (loop `get_all_heads_vert_scores`, score at the encoding index) + per-head stego-vs-cover stat test (FDR).

## Model decision
**Use Qwen-14B R1-distill.** Hooks are written and tested for Qwen2 natively, `model2layers_heads` is correct (48L/40H), and it's the stronger reasoner (matters given NEST's capability concern). Llama-3.1-8B would need a cloned `apply_llama_attn_mask_hooks` (doable — same GQA/RoPE/module names — but extra risk). Only fall back if VRAM-bound.
- **Top risk:** the suppression hook monkeypatches a specific `modeling_qwen2` forward signature → **transformers version drift can break it.** Pin to the repo's `requirements.txt`.

## Build order (smallest validating step first)
- **Phase 0 — hook sanity:** confirm the patched `masked_forward` reproduces stock logits with an empty mask. (De-risks the version issue — the biggest unknown.)
- **Phase 1 — task + scoring + capability gate:** generate Plain/Stego/Cover at small k (2–3), measure stego vs plain accuracy. **If stego accuracy collapses → decide fine-tuning here, cheaply, before any whitebox work.**
- **Phase 2 — correlational attention (A & B):** attention-to-encoding-token, receiver-head discovery, per-head stego-vs-cover test.
- **Phase 3 — causal suppression (C):** the double dissociation (hidden-task accuracy drops, cover survives).
- **Phase 4 — covertness monitor + write-up;** state the weakest supported claim (existence on one model/scheme).

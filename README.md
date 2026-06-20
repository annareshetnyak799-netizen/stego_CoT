# Steganographic CoT Detection via Activation Probing

Research on detecting steganographic chain-of-thought (CoT) reasoning in large language models using internal activations rather than text-level analysis.

## Research Question

Can a model's internal activations reveal steganographic computation - where the model hides its true reasoning inside innocent-looking CoT text - even when text-level monitors cannot detect it?

## Motivation

CoT monitoring is a foundational AI safety technique: if a model must show its reasoning, misaligned behavior becomes visible and correctable. This oversight breaks down if a model learns to encode its actual reasoning covertly while producing plausible-looking surface text. Unlike text-level steganalysis, activation-based detection could catch hidden computation *before* the first CoT token is written.

## Approach

Contrastive design: same task, two conditions — open reasoning vs. stego-instructed reasoning (acrostic: first letter of each CoT step encodes a payload word). All main experiments run on Llama-3.1-8B-Instruct; data collected in exp01 (303 contrastive pairs, 87% fidelity).

**Phase 1 — Activation probes (exp00–exp03):** linear probes on residual stream at pre-generation, mean-CoT, and post-generation readout points. Blocked by three compounding confounds (prompt, generation, weight). Established confound taxonomy; probe AUROC reflects prompt token differences, not steganographic computation.

**Phase 2 — Mechanistic attention analysis (exp04–exp07):** measured attention elevation at acrostic positions (sentence-start tokens). Found significant elevation (t>30, p<0.0001), concentrated in head 28 (L27) and a cluster at L11–12, but distributed across all 32 heads. Open acrostic control (exp06) produced 95.8% of stego elevation — falsifying the stego-specific interpretation: elevation is a structural property of any letter constraint.

**Phase 3 — Log-probability calibration (exp08–exp08b):** teacher-forcing log P at sentence-start positions. Found large opposing anomaly: within-sequence z-score is +0.637 in stego vs −0.470 in open (t=10, p≈0). Open acrostic control (exp08b) produced z=+0.514, nearly identical to stego — the same structural falsifier applies.

## Repository Structure

```
stego_CoT/
├── notebooks/
│   ├── exp00_pipeline_sanity_check.ipynb   # activation probe pipeline
│   ├── exp02b_probe_valid_pairs.ipynb       # linear probe on exp01 data
│   ├── exp03_model_organism.ipynb           # LoRA model organism
│   ├── exp04_attention_elevation.ipynb      # attention at acrostic positions
│   ├── exp04b_acrostic_vs_control_pos.ipynb # acrostic vs mid-sentence positions
│   ├── exp04c_per_head_heatmap.ipynb        # per-head attention heatmap
│   ├── exp05a_head_ablation.ipynb           # analytical head ablation
│   ├── exp06_control.ipynb                  # open acrostic control (attention)
│   ├── exp07_prompt_attention.ipynb         # prompt attention analysis
│   ├── exp08_logprob_calibration.ipynb      # log P at sentence-starts (stego vs open)
│   └── exp08b_logprob_control.ipynb         # log P at sentence-starts (control vs open)
├── results/
│   ├── exp01/valid_pairs.json               # 303 contrastive pairs, Llama-3.1-8B
│   ├── exp04/ … exp08b/                     # per-experiment outputs and plots
│   └── exp00_layer_profile.png
├── docs/
│   ├── papers.md                            # annotated literature survey
│   └── SelfIE_rus.pdf                       # Russian translation of SelfIE paper
├── requirements.txt
└── README.md
```

## Quick Start

```bash
git clone https://github.com/annareshetnyak799-netizen/stego_CoT.git
cd stego_CoT
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/exp00_pipeline_sanity_check.ipynb
```

Set your HuggingFace token for model downloads:
```bash
export HF_TOKEN=your_token_here
```

## Experiments

| ID | Description | Key finding | Status |
|---|---|---|---|
| exp00a/b | Activation probe: Qwen2.5-0.5B + Llama-3.1-8B | Prompt confound (AUROC=1 at layer 0) | Done |
| exp01 | Data collection: 303 pairs, Llama-3.1-8B, acrostic | valid_pairs.json; fidelity 87% | Done |
| exp02 | Logit lens (colleague) | Letter commit at layer 29–32 of 32 | Done |
| exp02b | Linear probe on exp01 data | Probe = confound detector, not stego detector | Done |
| exp03 | LoRA model organism; activation probe | Weight confound; H1 not confirmed | Done |
| exp04 | Attention elevation at acrostic positions | Stego >> open, t>30, p<0.0001; signal from K=1 | Done |
| exp04b | Acrostic vs mid-sentence control positions | Elevation acrostic-specific; absent at +8-token offset | Done |
| exp04c | Per-head attention heatmap (32×32) | Head 28 (L27) strongest; cluster L11–12 | Done |
| exp05a | Analytical head ablation (6 configurations) | Signal distributed; removing top-5 leaves 86% elevation | Done |
| exp06 | Open acrostic control — attention | Control = 95.8% of stego → **structural, not stego-specific** | Done |
| exp07 | Prompt attention analysis | Stego ≈ control in both metrics → structural | Done |
| exp08 | Log P at sentence-starts: stego vs open | z_stego=+0.637, z_open=−0.470; t=10, p≈0 | Done |
| exp08b | Log P at sentence-starts: control vs open | z_control=+0.514 ≈ z_stego → **structural, not stego-specific** | Done |

### Summary of findings

Both detection signals studied — attention elevation and log-probability elevation at constrained positions — are structural properties of any letter constraint, not diagnostic for steganographic hiding. A model following an open "start each step with S, A, F, E" instruction produces the same signal as a model given a hidden acrostic instruction.

The anomaly is genuine and large: in the stego/control conditions, sentence-start tokens are the most probable tokens in their CoT (z=+0.6), while in the open condition they are among the least probable (z=−0.5). This reflects instruction-following confidence, not steganographic intent.

## Related Work

| Paper | Key finding | Relevance |
|---|---|---|
| [Hide and Seek in Embedding Space](https://arxiv.org/abs/2601.22818) (Westphal et al., 2026) | Linear probes on later layers detect exfiltration stego | Closest prior work; we extend to CoT reasoning stego |
| [Detecting Strategic Deception via Linear Probes](https://arxiv.org/abs/2502.03407) (Goldowsky-Dill et al., 2025) | AUROC 0.96-0.999 for deception on Llama-3.3-70B | Probe methodology template |
| [Catching Rationalization in the Act](https://arxiv.org/abs/2603.17199) (Mirtaheri & Belkin, 2026) | Pre-gen probes predict motivated reasoning; best layer ~28 | Pre-generation readout design |
| [NEST](https://arxiv.org/abs/2602.14095) (Karpov, 2026) | 28-model eval; Claude Opus 4.5 achieves 92% on counting stego | Stego capability baseline |
| [Towards Safeguarding LLM FT APIs](https://arxiv.org/abs/2508.17158) (Youstra et al., 2025) | Probe monitors: 99%+ detection, generalizes to unseen ciphers | Cross-scheme generalization target |

Full annotated bibliography: [docs/papers.md](docs/papers.md)

## Novelty Over Prior Work

[Hide and Seek (2601.22818)](https://arxiv.org/abs/2601.22818) uses white-box probing of the attacker's own model for data exfiltration only. We extend in three directions:

1. **Passive-warden detection** - probe an external monitor model, not the attacker model
2. **CoT reasoning stego** - detect hidden reasoning, not just hidden data payloads
3. **Cross-scheme generalization** - probe trained on one encoding scheme tested on unseen schemes

## Status

Active research. Phases 1–3 complete (exp00–exp08b). Both attention-based and log-probability-based signals at acrostic positions have been characterized and found to be structural rather than stego-specific. Open questions: (1) whether any signal can distinguish hidden from open constraints; (2) propensity — whether models spontaneously produce steganographic structure without explicit instruction.

## Citation

If you use this work, please cite:

```bibtex
@misc{stego_cot_2026,
  title  = {Steganographic CoT Detection via Activation Probing},
  author = {Reshetnyak, Anna and collaborators},
  year   = {2026},
  url    = {https://github.com/annareshetnyak799-netizen/stego_CoT}
}
```

## License

MIT

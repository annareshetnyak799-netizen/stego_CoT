# Steganographic CoT Detection via Activation Probing

Research on detecting steganographic chain-of-thought (CoT) reasoning in large language models using internal activations rather than text-level analysis.

## Research Question

Can a model's internal activations reveal steganographic computation - where the model hides its true reasoning inside innocent-looking CoT text - even when text-level monitors cannot detect it?

## Motivation

CoT monitoring is a foundational AI safety technique: if a model must show its reasoning, misaligned behavior becomes visible and correctable. This oversight breaks down if a model learns to encode its actual reasoning covertly while producing plausible-looking surface text. Unlike text-level steganalysis, activation-based detection could catch hidden computation *before* the first CoT token is written.

## Approach

We probe the residual stream of open-weight LLMs (Llama-3.1-8B, Qwen2.5) to distinguish steganographic from open reasoning:

1. **Contrastive pairs** - same task, two conditions: open reasoning vs. stego-instructed reasoning
2. **Three readout points** - pre-generation (last prompt token), mean-CoT, post-generation
3. **Linear probes** - logistic regression + MLP per layer, scored by AUROC and recall@1%FPR
4. **Falsification gate** - transfer test across tasks and encoding schemes

The pre-generation readout tests whether "intent to hide" is already encoded in the residual stream before any CoT token is produced.

## Repository Structure

```
stego_CoT/
├── notebooks/
│   └── exp00_pipeline_sanity_check.ipynb  # activation probe pipeline (Tier A + Tier B)
├── results/
│   ├── exp00_layer_profile.png            # AUROC by layer (mean-CoT readout)
│   └── exp00_pre_vs_post.png             # pre vs. post-generation signal comparison
├── docs/
│   └── papers.md                          # annotated literature survey
├── requirements.txt
└── README.md
```

> **External dependency:** [Vector-SFT](https://github.com/Lucid-Layers-Inc/Vector-SFT) — a supervised fine-tuning implementation for inducing steganographic behavior in LLMs. Used as a reference for the stego model organism. Clone separately if needed.

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

| ID | Notebook | Description | Status |
|---|---|---|---|
| exp00 | `exp00_pipeline_sanity_check.ipynb` | Pipeline validation: Tier A synthetic + Tier B Qwen2.5-0.5B | Done |
| exp01 | *(in progress)* | Verified stego probe on Llama-3.1-8B, N=100+ pairs, encoding fidelity filter | Planned |
| exp02 | *(planned)* | Cross-scheme transfer: acrostic -> whitespace/synonym | Planned |

### exp00 Results (Tier A synthetic, N=200)

| Metric | Value |
|---|---|
| Best layer | 18 (of 33) |
| Logreg AUROC (mean-CoT) | 0.919 |
| Recall @ 1% FPR | 0.86 |
| Pre-gen AUROC | 0.770 |
| Transfer AUROC | 0.862 (PASS) |

Pre-generation AUROC > 0.5 confirms the hypothesis: steganographic intent is encoded in the residual stream before CoT generation begins.

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

Active research. Stage 1 (literature review + pipeline) complete. Stage 2 (Llama-3.1-8B experiments) in progress.

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

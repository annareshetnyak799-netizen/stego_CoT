# exp00a - Qwen2.5-0.5B-Instruct

**Date:** 2026-06-03
**Model:** Qwen/Qwen2.5-0.5B-Instruct
**Hardware:** Colab T4 GPU, bfloat16
**N:** 20 pairs (20 tasks x open/stego)

## Results

| Metric | Value |
|---|---|
| Best layer (mean-CoT) | 11 |
| Logreg AUROC | 1.000 |
| Recall @ 1% FPR | 1.00 |
| MLP AUROC | 0.963 |
| Pre-gen AUROC | 1.000 (layer 1) |
| Post-gen AUROC | 0.800 (layer 24) |
| Transfer AUROC | 0.722 |

## Interpretation

AUROC = 1.0 across all layers from layer 0 is a prompt artifact, not a computation signal.
Pre-gen AUROC = 1.0 at layer 1 confirms the probe detects different system prompts
(open vs stego), not steganographic computation in the residual stream.

Expected baseline result confirming the prompt confound.
exp01 will address this with encoding fidelity verification and prompt control.

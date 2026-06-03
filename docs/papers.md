# Papers

Our goal was to detect when a model hides its own reasoning from a monitor, by looking at its internal activations rather than just its text. While scoping that, I found two papers that were close but not exactly the same problem.

**TrojanStego** (Meier et al., EMNLP 2025, https://arxiv.org/abs/2505.20118) showed how a model could be fine-tuned to secretly leak information: it wrote a normal-looking answer while the word choices quietly encoded a hidden secret, recovered about 87 percent of the time. The difference from us: their hidden thing was a secret handed to the model in its prompt, so this was data exfiltration, not a model hiding its own reasoning. I saw it mainly as a way to build a hidden-behaviour model we could test against.

**Hide and Seek** (Westphal et al., 2026, preprint, https://arxiv.org/abs/2601.22818) built on it and tried to catch the hiding by probing the model's internal activations. A simple probe read the secret much better from a tampered model than a clean one, and that gap was their detection signal. The difference from us: they needed inside access to the attacker's own model and only tested the exfiltration case, not hidden reasoning. Still, the probing approach was close to what we had in mind, so I thought it could be a starting recipe, though it is a preprint and worth reproducing first.

More:

- **Eliciting Latent Knowledge from Quirky LMs** (Mallen, Belrose, COLM 2024) — peer reviewed, method is a plain linear probe. Code/model release not confirmed. https://arxiv.org/abs/2312.01037
- **Safeguarding Fine-tuning APIs against Cipher Attacks** (Youstra et al., 2025) — code repo exists, benchmark released, single linear probe on last-token activations. Repo not opened yet. https://arxiv.org/abs/2508.17158
- **Steganography via internal activations** (blog + code) — open 3B model, public data, runnable, but custom code and 0 stars/forks; authors say the reasoning wasn't actually hidden. https://github.com/Lucid-Layers-Inc/Vector-SFT
- **Eliciting Latent Knowledge with Mech Interp** (Taboo, 2025) — small proof of concept, open model, logit lens + SAEs. Code release not confirmed. https://arxiv.org/abs/2505.14352
- **Eliciting Secret Knowledge** (Cywiński et al., 2025) — must train the secret models plus an auditor first. https://arxiv.org/abs/2510.01070
- **Detecting Adversarial Fine-tuning with Auditing Agents** (2025) — needs fine-tuning-API access and frontier models. https://arxiv.org/abs/2510.16255
- **Auditing LMs for Hidden Objectives** (Marks et al., 2025) — built on a private model, hardest to match. https://arxiv.org/abs/2503.10965

explanation blogpost

README.md

00

01

02

03

04

05

06

lesswrong posts:

**artkpv**
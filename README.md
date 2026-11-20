# CropGuard-Q

**A quantum crop checker, stress-tested against tampering and quantum noise, with a real IBM hardware run.**

Entry for the *Moroccan Crop Classification Challenge* (Qiskit Fall Fest, EMSI × Quantum Africa). Classify fields as **alfalfa, maize or wheat** from satellite features, and answer one focused question:

> **Does adding a quantum model to a classical model add anything, especially when the data is tampered with or incomplete, compared with adding another classical model instead?**

Quantum computing (an 8-qubit kernel, exact simulation and two real IBM jobs), cybersecurity (an attack lab and a tamper alarm) and a bit of AI (the error-fixer, plus an optional analyst that writes incident notes). Project plan: [docs/PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md).

**Guide d'usage (français) : [docs/GUIDE.md](docs/GUIDE.md)** — installation, settings, console, notebook, and every function you need to use and understand.

No value is hard-coded: dataset facts come from `data/manifest.json`, study parameters from [`configs/experiment.json`](configs/experiment.json), selected settings from [`configs/cropguard.json`](configs/cropguard.json), and the simulated device from real IBM calibration data. `python -m challenge.report` rewrites the findings from the saved results.

## Results at a glance

All numbers come from [`notebooks/CropGuard_Q.ipynb`](notebooks/CropGuard_Q.ipynb) and the files in [`outputs/`](outputs/); this table is a snapshot, and `python -m challenge.report` prints the current findings from the saved results. Validation, 150 fields, seeds 0/1/2; settings were chosen on the same validation set, so scores are optimistic for every model alike.

| Model | Validation macro-F1 | Accuracy | Flags a human | Right when not flagged |
| --- | --- | --- | --- | --- |
| **Quantum committee** (submitted) | **0.921 ± 0.003** | 92.2% | 9.1% | 96.3% |
| Matched control (poly SVM replaces quantum) | 0.922 ± 0.003 | 92.2% | 8.4% | 96.4% |
| Official reference (RBF SVM, 13 features) | 0.880 | 88.0% | – | – |
| Quantum SVM alone / poly SVM / RBF SVM / forest | 0.920 / 0.927 / 0.933 / 0.904 | | | |

- **Clean data:** quantum committee minus matched control = −0.0004, 95% interval [−0.019, +0.024] (paired spatial-block bootstrap, 18 blocks, 2,000 resamples). No detectable difference. Both committees beat the reference by about 0.04, an interval that just touches zero.
- **Tampering (the 13 raw bands are edited; the attacker aims at the RBF SVM):** a total change of only 0.1 training SDs fools the RBF SVM silently on 24.9% of fields. With the full committee, 12.7% slip through (quantum) vs 14.7% (matched control). The quantum committee also flags more fields (19.1% vs 13.6%). Bootstrap: −2.0 points [−5.0, −0.2] at ε = 0.1, but no detectable difference at ε = 0.05 or 0.2. The partner that matters most is classical: at ε = 0.2 the forest cuts slips from 49% to 20%, the quantum SVM to 36%, the poly SVM to 40%. An attacker who knows all three still gets 50–55% through at ε = 0.5. Alone, the quantum SVM is not more robust: it degrades most with poisoned labels (0.78 at 30%, poly 0.85) and missing bands.
- **Noise + AI error-fixer (simulated device built from real `ibm_fez` calibration: median errors, gate times, readout, T1/T2):** a one-line linear regression cuts the similarity error from 0.052 to 0.011 at the device's own noise and from 0.131 to 0.015 at 3×, beating zero-noise extrapolation (0.032 / 0.094), which also costs twice the circuits. Realism check: the real `ibm_fez` job measured 0.073 raw, a bit worse than the median model predicts (per-qubit spread, crosstalk, drift). Not shown to improve final accuracy.
- **IBM hardware (inference only):**
  - `ibm_fez`, job `dc01nteq4uuc73a1tkq0`, submitted model (C = 100): the error-fixer, fitted on calibration pairs in the same job, cut kernel error from 0.073 to 0.018, yet only 1 of 4 quantum votes matched the exact simulator. Why: large C means large SVM weights, and they amplify small kernel errors. In simulation, 512-shot noise alone keeps only 49% of this model's votes.
  - `ibm_marrakesh`, job `dc01qrts2omc7388k9h0`, hardware-ready variant (C = 1, chosen offline as the largest C keeping ≥ 95% of votes under the measured error): **4 of 4 quantum votes matched**. On the tampered field, the raw hardware vote was wrong and the error-fixer brought it back, so the alarm fires on real hardware. One small inference example, not an accuracy estimate.

**What we do not claim:** no quantum advantage of any kind; the quantum model is not more robust than a classical one; the alarm does not stop an attacker who knows the defence; the error-fixer is not shown to improve accuracy; hardware runs are demos, not accuracy estimates.

## How it works

| Part | What it does |
| --- | --- |
| Quantum inspector | 8-qubit ZZ-feature-map fidelity kernel (exact statevectors, closed form checked against Qiskit) + SVM ([`challenge/kernel.py`](challenge/kernel.py)) |
| Classical inspectors | RBF SVM and random forest on the same 8 train-fitted PCA numbers |
| Vote + tamper alarm | majority label (RBF SVM settles a three-way split); flag when not unanimous ([`challenge/submission.py`](challenge/submission.py)) |
| Matched control | same pipeline, quantum SVM replaced by a polynomial SVM; every inspector tuned with the same 30-configuration grid ([`challenge/tune.py`](challenge/tune.py)) |
| Attack lab | raw-band tampering (naive and adaptive attackers), poisoned labels, missing bands, paired bootstrap ([`challenge/guard.py`](challenge/guard.py)) |
| AI error-fixer | linear regression noisy → exact kernel value; noisy-simulator study on a device built from real IBM calibration data ([`challenge/mitigation.py`](challenge/mitigation.py)) |
| IBM hardware | inference-only kernel circuits against the support vectors, with in-job calibration ([`challenge/hardware_kernel.py`](challenge/hardware_kernel.py)) |
| Console app | futuristic tkinter front-end: field scanner, tamper injector, kernel knob, attack lab, noise and QPU views ([`cropguard_app.py`](cropguard_app.py)) |
| AI analyst (optional) | writes a plain-text incident note for a flagged field; explains, never decides ([`challenge/analyst.py`](challenge/analyst.py)) |

`Submission` keeps the starter interface (`fit(X, y, seed)`, `predict`, `predict_proba`, `resources`) and adds `flag` and `inspect`. Its input is the 13 raw features in manifest order (B1…B12, NDVI); preprocessing is learned inside `fit` on training rows only and reproduces `data/views/pca8.npz` exactly.

## Install and run

Python 3.12 as pinned (also tested on CPython 3.15):

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

| Goal | Command | Time on an 8-core laptop |
| --- | --- | --- |
| Self-checks | `.venv/bin/python -m tests.test_cropguard` | ~10 s |
| Findings from the saved results | `.venv/bin/python -m challenge.report` | ~5 s |
| Re-tune (writes `configs/cropguard.json`) | `.venv/bin/python -m challenge.tune` | ~1 min |
| **Main notebook** (results, controls, extras) | `.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/CropGuard_Q.ipynb` | ~11 min (main run ~1 min) |
| **Console app** | `.venv/bin/python cropguard_app.py` | ~5 s to boot |
| Original quick demo | `.venv/bin/python demo.py` (needs `requirements-noise.txt`) | ~15 s |

The main run (notebook sections 1–4) needs no optional package, no network and no credentials, and takes about a minute. Optional extras:

- **Noise study:** `pip install -r requirements-noise.txt` (adds `qiskit-aer`). Without it the notebook shows the saved study.
- **IBM hardware:** `pip install -r requirements-hardware.txt`, then put your token in the environment only: `export QISKIT_IBM_TOKEN=...` (never in a file, notebook or commit). Set `RUN_HARDWARE = True` in the notebook, or press *Submit new inference job* in the console (it asks first). Each run of both jobs uses about 5 minutes of QPU time.
- **AI analyst with a language model:** the console writes an offline note by default. To let a model write it, point `CROPGUARD_LLM_URL` at any OpenAI-compatible chat endpoint, e.g. the free, keyless `https://text.pollinations.ai/openai`, Groq (`CROPGUARD_LLM_KEY`, `CROPGUARD_LLM_MODEL=llama-3.3-70b-versatile`) or a local Ollama. Only votes, labels and band changes are sent.

## Security notes

- Credentials live in environment variables only; `.gitignore` excludes `.env`, keys and `.qiskit/`. Hardware code needs `approved=True` and imports `qiskit_ibm_runtime` only inside the functions that use it.
- Inputs are validated at the boundary (`Submission` rejects non-finite or wrongly shaped inputs; `load_data` verifies every checksum).
- Language-model output is untrusted: shown as capped plain text, never executed, never fed back into a decision. Only `https` endpoints are accepted (plain `http` only for localhost).
- Attacks run on copies; the submitted model always trains on the clean released labels. Test labels are never used or recovered.

## Layout

- `challenge/`: organizers' starter (`data.py`, `models.py`, `classical.py`, `metrics.py`, `coverage.py`, `hardware.py`) plus our modules listed above, `config.py` (reads the settings files) and `report.py` (writes the findings from the results); `submission.py` is now the CropGuard-Q committee.
- `configs/experiment.json`: study parameters; `configs/cropguard.json`: frozen settings; `outputs/tuning_log.json`: every configuration tried; `outputs/device_profile.json`: real IBM calibration used by the noise study.
- `notebooks/CropGuard_Q.ipynb`: the deliverable notebook (executed, with outputs). The starter notebook is unchanged.
- `outputs/`: `main_results.json`, `attack_lab.json`, `noise_study.json`, `hardware_result.json`, `hardware_result_robust.json`, `hardware_robustness.json`, figures `cg_*.png`, and `predictions/` (validation predictions in the reference format; test predictions with flag and per-inspector votes; `quantum_committee_0_test.csv` is the submission file).
- `tests/test_cropguard.py`: self-checks. `cropguard_app.py`: console. `demo.py`: the original walkthrough.
- `data/`, `results/`: supplied data and reference runs, never edited.

## Rules we follow

- At most 8 qubits, exact local simulation for the main results; finite-shot, noisy and hardware runs are additional analysis.
- Every transformation and model is fitted on training data only; validation chooses settings; test is used only for final predictions from frozen settings. IDs and spatial-group codes are never inputs.
- The quantum SVM votes in every submitted prediction; the matched classical-only committee is the control, tuned with the same budget.
- Seeds 0, 1, 2 for stochastic parts (both SVMs are deterministic, and we say so); paired spatial-block bootstrap for every claimed difference.
- Hardware: inference only, with backend name, job ID and an exact-simulator comparison.

## Credits and licence

Built on the organizers' starter, [walid-mk/QiskitFallFest_EMSI_QuantumAfrica](https://github.com/walid-mk/QiskitFallFest_EMSI_QuantumAfrica) (copied at commit `510f862`), including its [student brief](https://github.com/walid-mk/QiskitFallFest_EMSI_QuantumAfrica/blob/main/Student_Challenge_Brief.pdf). Starter code and docs: [MIT](LICENSE). Data: adapted from Ouassanouan et al., [Moroccan Crop and Irrigation Dataset v2](https://doi.org/10.6084/m9.figshare.28486022.v2), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); see [data/README.md](data/README.md).

Methods: quantum kernels, Havlíček et al. 2019 ([arXiv:1804.11326](https://arxiv.org/abs/1804.11326)); classical–quantum ensembles, Grossi et al. 2022 ([arXiv:2208.07963](https://arxiv.org/abs/2208.07963)); shot noise and SVM robustness, Shastry et al. 2022 ([arXiv:2210.06971](https://arxiv.org/abs/2210.06971)); learned error mitigation, Czarnik et al. 2020 ([arXiv:2005.10189](https://arxiv.org/abs/2005.10189)) and Liao et al. 2023 ([arXiv:2309.17368](https://arxiv.org/abs/2309.17368)); benchmarking practice, Bowles et al. 2024 ([arXiv:2403.07059](https://arxiv.org/abs/2403.07059)); adversarial remote sensing, Czaja et al. 2018 ([arXiv:1805.10997](https://arxiv.org/abs/1805.10997)). The full list is in section 12 of the plan.

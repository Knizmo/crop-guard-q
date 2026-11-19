# CropGuard-Q

**A quantum crop checker that resists tampering and quantum noise.**

Entry for the *Moroccan Crop Classification Challenge* (Qiskit Fall Fest, EMSI × Quantum Africa). Classify fields as **alfalfa, maize or wheat** from satellite features, and answer one focused question:

> **Does adding a quantum model to a classical model add anything, especially when the data is tampered with or incomplete, compared with adding another classical model instead?**

Project plan and handoff: [docs/PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md) (teammates: sections 0 to 4 first, then 6, 7 and 10).

## The idea

| Part | What it does |
| --- | --- |
| 1. Quantum inspector | 8-qubit ZZ-feature-map fidelity kernel + SVM names the crop. |
| 2. Error-fixer (the only "AI correction") | A linear regression turns noisy similarity scores into exact ones, calibrated on field pairs whose exact score comes from classical simulation. Used only for noisy-simulator and hardware runs. |
| 3. Tamper alarm | Three inspectors (quantum SVM, classical RBF SVM, random forest, all on the same 8 numbers) classify each field. If they are not unanimous, the field is flagged for a human. |

**Submitted prediction:** majority vote of the three classifiers; if all three differ, the classical SVM's label. The quantum SVM takes part in every prediction. The flag is extra output, not a replacement label. **Matched control:** the same ensemble with the quantum member replaced by a second classical model.

## Build plan

| Tier | Work |
| --- | --- |
| 1. Core model (essential) | Quantum-kernel SVM, classical SVM and forest, 3-model vote as the `Submission`, seeds 0/1/2, all metrics and resources |
| 2. Differentiator (essential) | Attack lab (tampering, poisoned labels, missing measurements), the alarm, matched classical-only ensemble, paired bootstrap |
| 3. Extra analysis | Noisy-simulator study and the error-fixer |
| 4. IBM hardware bonus (+5) | Inference-only job on a real device, compared with the exact simulator; backend name and job ID on the slides |

If time runs short, drop Tier 3 first and keep Tiers 1, 2 and the hardware bonus.

## Status (feasibility tests, simulation only, validation set, indicative)

- Validation macro-F1: classical RBF SVM 0.932, quantum ZZ-kernel SVM 0.913 (scale 0.1), random forest 0.893, official reference 0.880, starter QNN 0.856. The kernel is very sensitive to the rotation strength: scale 1.0 collapses to ≈ 0.23–0.37.
- The quantum model is **not** more robust than the classical one against tampering, label poisoning or missing measurements.
- The alarm cuts undetected attacks from 41% (classical alone) to 28% (+ quantum), 23% (+ forest) and 20% (+ both), flagging 6–11% of clean fields. A quantum partner is no better than a classical one, and an attacker who knows the whole committee still gets through 27–31%.
- The error-fixer lowers similarity error from 0.081 to 0.013 (3× noise: 0.137 to 0.037), beating zero-noise extrapolation (0.059). It is not shown to improve final accuracy.

Final claims must come from full runs on seeds 0, 1, 2. See section 11 of the plan for what must **not** be claimed (no quantum advantage of any kind).

## Install and run

Python 3.12:

```bash
git clone https://github.com/Knizmo/crop-guard-q.git
cd crop-guard-q
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name crop-challenge --display-name "Crop challenge"
```

**Quick demo (~15 s):** quantum kernel vs classical SVM, a tamper alarm under attack, and the error-fixer on a simulated noisy device. It predates the plan: its alarm uses two models (quantum + classical SVM), not the three-model vote.

```bash
.venv/bin/python -m pip install -r requirements-noise.txt
.venv/bin/python demo.py
```

**Starter notebook:** open `notebooks/Qiskit_QuantumAfrica_Crop_Challenge.ipynb`, select the **Crop challenge** kernel and run all cells. The main run needs no IBM token and no optional package. Only the hardware bonus needs `requirements-hardware.txt` and `QISKIT_IBM_TOKEN` in your environment (never in a file or commit).

## Layout

- `demo.py`: the CropGuard-Q walkthrough; `challenge/kernel.py`: our 8-qubit quantum kernel + SVM.
- `challenge/`: starter code (`models.py` encoding/QNN/control, `submission.py` final `fit`/`predict` interface, `coverage.py`, `data.py`, `metrics.py`, `hardware.py`).
- Planned (plan section 8): `challenge/submission.py` becomes the 3-model vote with flag; `challenge/guard.py` attack lab and alarm metrics; `challenge/mitigation.py` noisy simulation and calibration; `challenge/hardware_kernel.py` IBM inference for the kernel; `notebooks/CropGuard_Q.ipynb` main notebook.
- `outputs/`: our predictions, tables, figures and `hardware_result.json`.
- `data/`, `results/`: supplied data and reference runs; never edit them (`data/` is checksummed in `manifest.json`). See [data/README.md](data/README.md).
- `docs/`: [project plan](docs/PROJECT_SUMMARY.md), plus the organizers' [evaluation guide](docs/EVALUATION.md), [hints](docs/HINTS.md) and [coverage commands](docs/COVERAGE.md).

## Rules we follow

- At most 8 qubits with exact local simulation for the main results; finite-shot and hardware runs are additional analysis.
- Every transformation and model fitted on training data only; validation for choosing settings, test only for final predictions from frozen settings. IDs and spatial-group codes are never inputs.
- The quantum outputs take part in the submitted prediction; a matched classical-only ensemble is the control.
- Seeds 0, 1, 2 for stochastic parts (both SVMs are deterministic, and we say so).
- Report macro-F1, accuracy, per-class precision/recall, confusion matrices, training and prediction time, support vectors, kernel storage, qubits, depth, circuit evaluations, and the number of configurations tried.
- Any claimed difference between models uses a paired bootstrap over spatial blocks (`data/validation_groups.json`).
- Hardware bonus: inference only, with backend name, job ID and an ideal-simulator comparison.

## Credits and licence

Built on the organizers' starter, [walid-mk/QiskitFallFest_EMSI_QuantumAfrica](https://github.com/walid-mk/QiskitFallFest_EMSI_QuantumAfrica) (copied at commit `510f862`), including its [student brief](https://github.com/walid-mk/QiskitFallFest_EMSI_QuantumAfrica/blob/main/Student_Challenge_Brief.pdf), which is not copied here. Starter code and docs: [MIT](LICENSE). Data: adapted from Ouassanouan et al., [Moroccan Crop and Irrigation Dataset v2](https://doi.org/10.6084/m9.figshare.28486022.v2), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); see [data/README.md](data/README.md) for attribution and modifications. Method references are in section 12 of the plan. The EMSI and Quantum Africa logos and the dataset-paper figure are not included.

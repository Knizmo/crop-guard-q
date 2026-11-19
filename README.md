# CropGuard-Q

**A quantum crop checker that resists tampering and quantum noise.**

Entry for the *Moroccan Crop Classification Challenge* (Qiskit Fall Fest, EMSI × Quantum Africa). Classify fields as **alfalfa, maize or wheat** from satellite features, and ask the question a real crop-insurance checker must answer: *what if the data was tampered with, and can we detect it?*

Full write-up: [docs/PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md).

## The idea

| Part | What it does |
| --- | --- |
| 1. Quantum inspector | 8-qubit quantum kernel (field-to-field similarity) + SVM names the crop. |
| 2. AI noise-canceller | A small model learns to map noisy kernel values to exact ones, calibrated on ~300 classically simulated pairs. |
| 3. Tamper alarm | A quantum and a classical model classify each field; if they disagree it is flagged for human review. |

## Status (feasibility tests, simulation only, validation set, indicative)

- Quantum kernel + SVM reaches macro-F1 ≈ 0.91 (classical SVM on the same 8 features: 0.93; official reference: 0.88; starter QNN: 0.86). The kernel is very sensitive to the data-rotation strength: the default collapses to ≈ 0.2–0.4, 0.05–0.1 works.
- The quantum model is **not** more robust than the classical one against tampering, label poisoning or missing data. We report this honestly and claim no quantum advantage.
- The tamper alarm roughly halves silent attacks (41% → 20%) at the cost of flagging 6–11% of clean fields.
- The AI noise correction gives ~6× lower kernel error than raw noisy results; a linear fit is enough. It is not yet shown to improve final accuracy.

Final claims must come from full runs on seeds 0, 1, 2 and the test set.

## Install and run

Python 3.12:

```bash
git clone https://github.com/Knizmo/crop-guard-q.git
cd crop-guard-q
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name crop-challenge --display-name "Crop challenge"
```

**Quick demo (~15 s):** the three CropGuard-Q parts end to end: quantum kernel vs classical SVM, the tamper alarm under attack, and the noise-canceller on a simulated noisy device.

```bash
.venv/bin/python -m pip install -r requirements-noise.txt
.venv/bin/python demo.py
```

**Starter notebook:** open `notebooks/Qiskit_QuantumAfrica_Crop_Challenge.ipynb`, select the **Crop challenge** kernel and run all cells. Local simulation needs no IBM token. Only the optional hardware run needs `requirements-hardware.txt` and credentials in your environment.

## Layout

- `demo.py`: the CropGuard-Q walkthrough; `challenge/kernel.py`: our 8-qubit quantum kernel + SVM.
- `challenge/`: starter code (`models.py` encoding/QNN/control, `submission.py` final `fit`/`predict` interface, `coverage.py`, `data.py`, `metrics.py`).
- `data/`: supplied tables and PCA views; keep unchanged (checksummed in `manifest.json`). See [data/README.md](data/README.md).
- `configs/`, `results/`: starter configuration and reference runs. Save our own runs separately.
- `docs/`: [project summary](docs/PROJECT_SUMMARY.md), plus the organizers' [evaluation guide](docs/EVALUATION.md), [hints](docs/HINTS.md) and [coverage commands](docs/COVERAGE.md).

## Rules we follow

At most 8 qubits with exact local simulation; transformations fitted on training data only; validation for choosing settings, test only for final predictions; seeds 0, 1, 2; report macro-F1, accuracy, per-class precision/recall, confusion matrices, runtime and model resources.

## Credits and licence

Built on the organizers' starter, [walid-mk/QiskitFallFest_EMSI_QuantumAfrica](https://github.com/walid-mk/QiskitFallFest_EMSI_QuantumAfrica) (copied at commit `510f862`), including its [student brief](https://github.com/walid-mk/QiskitFallFest_EMSI_QuantumAfrica/blob/main/Student_Challenge_Brief.pdf), which is not copied here. Starter code and docs: [MIT](LICENSE). Data: adapted from Ouassanouan et al., [Moroccan Crop and Irrigation Dataset v2](https://doi.org/10.6084/m9.figshare.28486022.v2), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); see [data/README.md](data/README.md) for attribution and modifications. The EMSI and Quantum Africa logos and the dataset-paper figure are not included.

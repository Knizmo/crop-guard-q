# CropGuard-Q: project plan and handoff

**Moroccan Crop Classification Challenge · Qiskit Fall Fest · EMSI × Quantum Africa**

Repository: <https://github.com/walid-mk/QiskitFallFest_EMSI_QuantumAfrica>

## How to use this file

- **Teammates:** read sections 0 to 4 first (written for beginners), then 6, 7 and 10.
- **Claude Code:** read the whole file. Section 8 holds your instructions, section 9 holds tested code, section 7 covers the IBM hardware bonus, and section 11 lists what must not be claimed. Start with Tier 1.
- Suggested prompt for Claude Code, after saving this file in the repo root as `PROJECT_PLAN.md`:
  > Read PROJECT_PLAN.md completely. Follow section 8. Build Tier 1 first, verify it in a fresh environment, then report the numbers before starting Tier 2.

---

## 0. Task checklist (everything that must be done)

**A. Setup**
- [ ] Create a Python 3.12 environment and install `requirements.txt`
- [ ] Run the organizers' starter notebook once and confirm the baseline numbers
- [ ] Request an IBM Quantum account and API token (needed for the bonus, see section 7)
- [ ] Ask the organizers for the submission route, whether hardware use needs approval, and the judging-machine details

**B. Tier 1: the core model (essential)**
- [ ] Quantum-kernel SVM on the 8-number view (8 qubits, exact simulation)
- [ ] Tune the rotation-strength knob and the SVM settings on validation, then freeze them
- [ ] Classical SVM and random forest on the same 8 numbers (comparison models)
- [ ] Final `Submission`: 3-model vote where the quantum model takes part
- [ ] Run seeds 0, 1, 2 and produce all required metrics and resource numbers

**C. Tier 2: what makes the project stand out (essential)**
- [ ] Matched classical-only ensemble (replace the quantum member with another classical model)
- [ ] Attack lab: tampering, poisoned labels, missing measurements
- [ ] Tamper alarm: flag a field when the three models disagree
- [ ] Paired bootstrap for any claimed difference between models

**D. Tier 3: extra analysis**
- [ ] Noisy-simulator study and the AI error-fixer (linear regression)

**E. Tier 4: IBM hardware bonus (+5), do not forget**
- [ ] Inference-only job on a real IBM device, compared with the ideal simulator (see section 7)
- [ ] Backend name, job ID and comparison recorded in the notebook and on the slides

**F. Deliverables**
- [ ] One runnable notebook or code package, with a short execution instruction
- [ ] 3 to 5 slides
- [ ] Citations for reused code and methods
- [ ] Submit through the route in the repository README

---

## 1. The challenge in brief

**Task.** Classify fields as **alfalfa, maize or wheat** from satellite measurements (Doukkala region, Morocco). Build a quantum or hybrid model, compare it with the supplied classical reference, and explain what the experiments show. A result below the reference can still earn credit if the comparison and conclusions are well supported.

**Data** (all in `data/`, never modify these files, they are checksum-verified):

| Split | Fields | Labels |
|---|---|---|
| train | 914 | yes |
| validation (called "development" in the PDF) | 150 (50 per crop) | yes |
| test | 221 | hidden, used for final predictions only |

- Raw features, 13 numbers per field: Sentinel-2 bands `B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B11 B12` and `NDVI`.
- Ready-made compressed views (PCA fitted on training data only): `data/views/pca4.npz` and `data/views/pca8.npz`. Keys: `labels, X_train, ids_train, y_train, X_validation, ids_validation, y_validation, X_test, ids_test`.
- `data/validation_groups.json` maps each validation `sample_id` to a spatial block, for paired bootstrap comparisons. IDs and group codes are **never** model inputs.
- Training classes are imbalanced. Validation is balanced. Labels are crop names (strings) in the order `('alfalfa', 'maize', 'wheat')`.

**Official starting results** (validation, all 914 training fields, seeds 0/1/2):

| Model | Mean macro-F1 | Mean accuracy |
|---|---|---|
| Classical baseline, RBF SVM on all 13 features | 0.8797 | 88.00% |
| Starter 4-qubit QNN | 0.8557 | 85.56% |
| Small classical control (11 parameters) | 0.5814 | 65.11% |

The starter's classical control is tiny, so it is a weak comparison. We do not rely on it.

**Rules that matter** (from the PDF, `README.md`, `docs/EVALUATION.md`, `docs/HINTS.md`):

1. Fit every transformation on **training data only**. Use validation for choosing settings. Use test only for final predictions. Keep the supplied split unchanged.
2. The main evaluation is **exact local simulation**, **at most 8 qubits per circuit**. The README publishes a runtime budget (up to 15 minutes per starter seed and 35 minutes per final seed, including preprocessing, fitting, feature generation, cache construction and prediction; judging-machine details are still to be announced). Keep the main run light and put heavy extras in separate, optional cells.
3. Choose **one focused question** (or a few). One clear, controlled experiment is enough.
4. Use a **matched classical control** with the same input features and preprocessing. For a hybrid model, replace or remove the quantum part and keep the shared classical processing.
5. Run **seeds 0, 1, 2** for the selected model and stochastic controls. Report mean and standard deviation. A deterministic model may reuse its identical result if that fact is stated.
6. Report: **macro-F1, accuracy, per-class precision and recall, confusion matrix, training and prediction time, parameter counts or equivalent costs** (for SVMs and trees: support-vector counts, tree sizes, kernel storage, runtime), and **quantum resources** (qubits, depth, circuit evaluations). State roughly how many configurations were tried and how much tuning effort was used.
7. Any claim about a difference between models should use **paired field or spatial-block resampling**, and state the method.
8. **The quantum outputs must participate in the submitted prediction workflow.**
9. Hardware is optional (**+5 bonus**): verified IBM **inference only** (no training on hardware). Provide backend name, job ID and a brief ideal-simulator comparison. Finite-shot and hardware runs are additional analysis, separate from the main exact result.
10. Submit **one runnable notebook or code package** (model, matched control, configurations, results, short execution instruction) plus **3 to 5 slides**. Organizers re-run the workflow. Cite reused code and methods.

**Scoring** (two judges rate the last two categories from 0 to 5; ties go to experimental quality, then performance):

| Criterion | Points | Basis |
|---|---|---|
| Predictive performance | 30 | 30 × mean final-test macro-F1 |
| Experimental quality and analysis | 40 | controls, one focused comparison, resource measurements, supported interpretation. Clear negative findings can earn full credit. |
| Model design and reproducibility | 30 | justified choices, working implementation, clear explanation |
| Hardware bonus | +5 | verified IBM inference |

---

## 2. Our idea in plain words

### 2.1 The problem we frame

Satellite crop checks are used in practice, for example to verify insurance or subsidy claims (NASA reports Landsat imagery being used in crop-insurance fraud investigations). A checker like this is only useful if it can be trusted. If someone alters a field's measurements, the model may give the wrong crop. Quantum hardware also adds errors of its own.

### 2.2 Headline question (the single focused question for the judges)

> **Does adding a quantum model to a classical model add anything, especially when the data is tampered with or incomplete, compared with adding another classical model instead?**

This sits inside the brief: it is the "classical and quantum ensemble" approach from `docs/HINTS.md`, and it extends the hint questions "missing measurements" and "sampling". Everything else (attack lab, noise study, hardware run) serves this one question.

### 2.3 The parts

| Part | What it does | Everyday picture |
|---|---|---|
| **1. Quantum inspector** | An 8-qubit circuit computes a similarity score between two fields (a quantum kernel). An SVM uses the table of scores to name the crop. | A detective comparing a new field with known fields |
| **2. Error-fixer (the only "AI correction")** | A linear regression learns to turn noisy similarity scores into ideal ones, using field pairs whose exact score is known from classical simulation. | Calibrating a scale that always reads 2 kg too much |
| **3. Tamper alarm** | Three models classify each field independently. If all agree, the field is "verified". If any disagree, it is flagged for a human. | Two or three inspectors double-checking each other |
| **Test lab (not a model)** | Code that tampers with copies of the data to measure how often each model is fooled. | A crash-test rig |

**Important:** the alarm fires when the models **disagree**, not when they agree.

### 2.4 The four models

| # | Model | Role |
|---|---|---|
| 1 | **Quantum SVM**: ZZ-feature-map fidelity kernel on 8 qubits, then an SVM with a precomputed kernel | Classifies. This is our quantum model. |
| 2 | **Classical SVM**: RBF kernel on the same 8 numbers | Classifies. Also our fair comparison for model 1. |
| 3 | **Random forest** on the same 8 numbers | Classifies. Third voter, with a different inductive bias. |
| 4 | **Linear regression** (noisy score to ideal score) | Corrects numbers. Not a classifier. Used only for noisy-simulator and hardware runs. |

Models 1 to 3 are the "inspectors". Only model 4 is the "AI correction".

### 2.5 What we submit as predictions

- **Final label** for each test field: majority vote of the three classifiers. If all three differ, use the classical SVM's label. The quantum SVM therefore takes part in every prediction.
- **Flag**: raised when the three are not unanimous. It is extra output for the analysis, not a replacement for the label (every test field still gets a label).
- **Matched control ensemble** (required by the brief): the same pipeline with the quantum member replaced by a second classical model of comparable tuning effort (for example an SVM with a different kernel, or a small MLP). Compare the two ensembles on identical inputs, splits and seeds.

Freeze all settings before producing test predictions. We never see test labels.

---

## 3. Quantum crash course (no maths)

- **Qubit.** A bit is a switch: 0 or 1. A qubit is more like a dial. Its position sets the odds of reading 0 or 1.
- **Gate.** Turning a dial. A **circuit** is a list of dial-turns. We use a field's numbers as the angles.
- **Measurement.** Reading a dial gives only 0 or 1, with odds set by the dial. We repeat it many times (called **shots**, for example 1024) and count the results to estimate the odds.
- **Entanglement.** Linking qubits so turning one changes how the others behave. It makes our similarity score richer.
- **Quantum kernel.** To compare fields A and B: apply A's circuit, then undo B's circuit, and measure how often we return to the starting state. That probability (0 to 1) is the **similarity score**. Computing it for all pairs gives a table the SVM learns from.
- **Exact simulation.** With 8 qubits, a normal computer can compute the exact result, so we do not need a real quantum machine for the main results. Real machines add noise.
- **Noise.** Imperfect gates, qubits losing their state and misreads at the end. In our tests the noisy score behaved roughly like `measured ≈ a · true + b` (a gain and an offset), which is why a straight-line fit can correct it.

**The knob.** The kernel has one key setting: how strongly the data rotates the qubits (a scale factor on the features). Too strong, and every field looks different from every other (kernel collapse). Small values work.

---

## 4. Tools

| Job | Tool | What it does here |
|---|---|---|
| Language | **Python 3.12** | Everything. The repo is pinned to CPython 3.12. |
| Environment | **venv + pip**, `requirements.txt` | Isolated install of the exact pinned versions |
| Notebook | **Jupyter** (`ipykernel`, `nbconvert`) | The notebook is our deliverable. `nbconvert --execute` re-runs it headlessly. |
| Data | **pandas, NumPy, SciPy** | Tables, fast math |
| Plots | **Matplotlib** | Charts for the slides |
| Machine learning | **scikit-learn 1.9.1** | `SVC` (accepts a precomputed kernel), `RandomForestClassifier`, `LinearRegression`, metrics, PCA |
| Quantum | **Qiskit 2.4.2** | Circuits, `zz_feature_map`, `Statevector` for exact simulation |
| Quantum ML | **qiskit-machine-learning 0.9.1** | Used by the organizers' starter QNN. We keep it only as a comparison. |
| Noisy simulation (optional) | **qiskit-aer** | Imitates a noisy device. **Not** in `requirements.txt`, so it must be an optional extra. |
| Hardware (optional, bonus) | **qiskit-ibm-runtime** (`requirements-hardware.txt`), IBM Quantum account and token | Sends circuits to IBM and returns counts. Token goes in the environment variable `QISKIT_IBM_TOKEN`, never in the notebook. |
| Organizers' helpers | `challenge/` (`data.py`, `models.py`, `classical.py`, `metrics.py`, `submission.py`, `coverage.py`, `hardware.py`) | Data loading with checks, starter models, scoring, submission interface (`fit(X, y, seed)`, `predict`, `predict_proba`, `resources`), training-size runs, hardware inference helper |
| Slides | Any tool (PowerPoint, Canva, Google Slides) | 3 to 5 slides |

**Compute needs are small.** Everything below ran on 2 CPU cores with no GPU. A laptop with 8 GB of RAM is enough.

---

## 5. Evidence from quick feasibility tests

**Read this with care.** These were quick tests: exact simulation, one validation split of 150 fields (about ±3 points of uncertainty), settings chosen on that same validation set (so numbers are optimistic), simple attacks, and simulated noise. The final claims must come from the full 3-seed runs and the held-out evaluation. Use these numbers to plan, not to claim.

### 5.1 Clean accuracy (macro-F1, validation, pca8 view unless noted)

| Model | Macro-F1 |
|---|---|
| Classical RBF SVM on pca8 (`C=10, gamma='scale'`) | 0.932 |
| Quantum ZZ-kernel SVM (scale 0.1, `C=10`) | 0.913 |
| Random forest (300 trees) | 0.893 |
| Official reference (all 13 features) | 0.880 |
| Starter QNN (4 qubits) | 0.856 |

The 8-component view carries more signal than the 4-component view the starter uses (classical SVM on pca4: about 0.85).

**Kernel-collapse finding.** On pca8 the ZZ kernel with scale 1.0 gave macro-F1 of only about 0.23 to 0.37 (1 or 2 repetitions). With scale 0.05 to 0.1 it gave about 0.89 to 0.91. Deeper (2-repetition) circuits and larger scales were worse. A projected quantum kernel (RBF on single-qubit Bloch vectors) and a hybrid kernel (quantum plus RBF) did not beat the plain RBF SVM in quick tests, but neither was explored thoroughly.

### 5.2 Attacks (all on copies of the validation or training data)

Tampering: random-search attack, L2 budget ε in standardized-PCA units (the typical distance between two fields is roughly 3). Accuracy after an attack aimed directly at that model:

| ε | Classical → classical | Quantum → quantum |
|---|---|---|
| 0.25 (small) | 0.80 | 0.79 |
| 0.5 (medium) | 0.59 | 0.60 |
| 1.0 (large) | 0.23 | 0.22 |

Transfer (attack crafted on the classical SVM, accuracy of another model): ε=0.25: quantum 0.81, forest 0.82. ε=0.5: 0.66 and 0.73. ε=1.0: 0.36 and 0.46.

Poisoned labels (random flips in training, mean of 5 draws) and missing features (k of 8 set to 0, mean of 20 draws), macro-F1:

| Setting | Classical | Quantum |
|---|---|---|
| 10% wrong labels | 0.913 | 0.884 |
| 20% wrong labels | 0.883 | 0.864 |
| 30% wrong labels | 0.835 | 0.827 |
| 2 of 8 missing | 0.786 | 0.789 |
| 4 of 8 missing | 0.617 | 0.637 |

**Main finding: the quantum model is not more robust than the classical one.** Both are about equally fragile.

### 5.3 The alarm (ε = 0.5, attacker aims at the main classical model only)

| Setup | Attack succeeds undetected | Clean fields flagged | Accuracy on unflagged clean fields |
|---|---|---|---|
| Classical SVM alone | 41% | 0% | 93% |
| + quantum partner | 28% | 6% | 95% |
| + forest partner | 23% | 7% | 94% |
| + both partners | 20% | 11% | 96% |

"Succeeds undetected" means all models in the committee output the same wrong crop. At ε=0.25 the single model is fooled 20% of the time, and the committees let 14 to 17% through.

**Adaptive attacker** (knows the whole committee and minimizes the largest margin): 31%, 31% and 27% get through at ε=0.5 (17%, 17%, 15% at ε=0.25). The alarm helps against naive attacks only.

**A quantum partner is not better than a classical partner.** This is why the matched classical-only ensemble is a required control.

### 5.4 Noise study (simulated noisy device)

Setup: Qiskit Aer density-matrix simulation, 60 training and 45 validation fields, scale 0.1, 1024 shots, compute-uncompute circuits. Noise: 1-qubit depolarizing 0.2% on `h`, 2-qubit depolarizing 1.5% on `cx`, thermal relaxation (T1 100 µs, T2 80 µs, gate durations 50 ns and 400 ns), readout flips 3% and 4%. A second run used 3× all error rates. The AI correction was calibrated on 300 random training-pair entries and tested on pairs it never saw.

Mean absolute error of the similarity score versus the exact value (validation-versus-train pairs):

| Method | Normal noise | 3× noise | Circuit runs per pair |
|---|---|---|---|
| Raw noisy | 0.081 | 0.137 | 1 |
| Zero-noise extrapolation (folds 1 and 3, linear) | 0.059 | 0.128 | 2 |
| **AI correction (linear fit on the raw value)** | **0.013** | **0.037** | 1 |

A linear fit on raw plus folded values and boosted trees gave about the same error, so heavy AI is not needed.

**Not proven:** better scores did not clearly give better classification. On the 45-field subset, macro-F1 was 0.865 with the exact kernel, 0.887 raw noisy, 0.795 (linear) and 0.841 (trees) after correction. At 3× noise raw noisy scored 0.956 (above exact), probably noise acting as regularization, but 45 fields cannot separate these. Claim only "more accurate similarity values" unless a full-size run shows more.

Also note: the correction works because our circuits are small enough to simulate exactly for the calibration labels. Real devices drift, so the correction must be recalibrated for each session.

---

## 6. Build plan (tiers)

**Tiers 1 and 2 are essential. Tier 3 is extra analysis. Tier 4 is the IBM bonus.** Build in this order so something submittable exists early.

| Tier | Work | Acceptance criteria |
|---|---|---|
| **1. Core model** | Environment, starter notebook run, quantum-kernel SVM, classical comparison models, 3-model vote as the `Submission`, 3 seeds, resource and runtime reporting | Runs in a fresh venv from `requirements.txt`. Validation macro-F1 at or above the 0.88 reference (target about 0.90 or more). Tables for all required metrics. Main run well inside the published runtime budget. |
| **2. Differentiator** | Attack lab (tampering, poisoning, missing features), the alarm, matched classical-only ensemble, paired bootstrap | Tables like sections 5.2 and 5.3 produced for both ensembles on identical inputs. Bootstrap uses `validation_groups.json`. |
| **3. AI error-fixer (extra analysis)** | Noisy simulation on a subset, calibration on field pairs, correction, error-versus-noise plot | Optional import of `qiskit_aer`; the main notebook still runs without it. Plot and table like section 5.4. |
| **4. IBM hardware (bonus, +5)** | Small inference-only job on a real device, compared with the exact simulator | Backend name, job ID and comparison in the notebook and on the slides. Details in section 7. |

---

## 7. IBM hardware bonus (+5), do not forget

**What the bonus is.** Up to **+5 points beyond the 100-point rubric** for a verified IBM real-hardware demonstration. It is **inference only**: the model is trained locally, then a small part of the inference is run on a real device. It is optional, and the main evaluation stays simulation-based. Required evidence: **backend name, job ID and a brief ideal-simulator comparison**.

**Why it is worth doing.** +5 points is as large as a 0.17 jump in macro-F1, and the hardware run also gives us a real-device test of the quantum part.

### 7.1 Prerequisites

- An IBM Quantum Platform account and API key (token) with access to quantum hardware.
- Permission to use the chosen account and backend (the README says hardware use needs this). Ask the organizers if approval is required.
- `requirements-hardware.txt` installed (it adds `qiskit-ibm-runtime`). Keep this separate from the main environment, because the main run needs no credentials.
- The token in the environment variable `QISKIT_IBM_TOKEN`. **Never put the token in the notebook, a file, a commit or a slide.**

### 7.2 What to run on hardware

Our quantum part is the kernel, so hardware inference means measuring similarity scores between a field and the model's **support vectors**.

- For one field, the SVM needs one circuit per support vector (the count is `svm_q.n_support_.sum()`).
- Use a small number of **validation** fields for the demo (we know their true crop, so we can compare with it as well as with the simulator). Do not use hardware for test predictions.
- If the support-vector count is too large for your account's allowance, use a compact demo model (for example trained on a smaller subset, or with settings that give fewer support vectors) and **say so clearly**. The demo is one fixed-model inference example, not an accuracy estimate.
- Each circuit is the compute-uncompute circuit from section 9.6 with `fold=1`. Building it needs only Qiskit (Aer is needed only for simulators).

### 7.3 Steps

1. Train the quantum-kernel SVM locally and freeze it. Record the exact kernel values and the exact predicted crop for the demo fields (from section 9.2).
2. Test the whole hardware code path first on a local noiseless simulator, then (optionally) on the noisy simulator.
3. Choose a backend with enough qubits (8 needed). Check its status and queue, and prefer one with low reported error rates.
4. Transpile for that backend (`generate_preset_pass_manager`, fixed `seed_transpiler`, as in `challenge/hardware.py`). Record the transpiled **depth** and **two-qubit gate count**.
5. Start with a tiny smoke test (a few circuits), then submit the full batch as one job with 512 to 2048 shots. Print and save the **job ID**.
6. Compare with the exact simulator:
   - measured versus exact similarity score for each pair (mean absolute error, scatter plot)
   - the SVM decision value and the **predicted crop** from hardware values versus exact values (and versus the true crop)
7. Optional, if the circuit allowance permits: run a small calibration set of pairs on the same device in the same session, fit the error-fixer (section 9.6), and show the corrected hardware values next to the raw ones. This links Part 2 to a real device.
8. Save everything to `outputs/hardware_result.json`: backend name, job ID(s), shots, number of circuits, qubits, transpiled depth, comparison numbers.

### 7.4 How to keep it safe for the re-run

- In the notebook set `RUN_HARDWARE = False` by default. The hardware cell reads the **saved** `outputs/hardware_result.json` and displays it, so organizers can re-run the notebook without credentials.
- The main workflow must not import `qiskit_ibm_runtime` at the top level.
- Start from `challenge/hardware.py`, which already shows the working connection, transpile and run pattern for the pinned environment (including the `approved=True` guard). It is written for the starter QNN, so adapt it to the kernel circuits.

### 7.5 What goes on the slides

On the last slide: the backend name, the job ID, a one-line comparison with the simulator (for example "measured scores within X of exact, predicted crop matches"), and the limitation: one small fixed-model inference example, not an accuracy estimate.

### 7.6 Pitfalls

- Training on hardware is not allowed for the bonus.
- A job ID alone is not enough: include the simulator comparison.
- Hardware noise will shift the scores. Report the difference honestly. That is a result, not a failure.
- Do not extend the result to a claim about accuracy or quantum advantage.

---

## 8. Instructions for Claude Code

**Goal.** Build the project above inside a copy of the repo, verified and explainable by beginners.

**Hard constraints**
1. Never edit `data/` or `results/` (reference results stay as they are). Write our outputs to `outputs/`.
2. Fit scalers, PCA, imputers and every model on **training data only**. Use validation to choose settings. Produce test predictions only from frozen settings. Never use `sample_id` or spatial-group codes as features. Never try to recover test labels.
3. Main results: exact simulation, **at most 8 qubits**. No network access needed for the main run.
4. Attacks, poisoning and missing-feature tests run on **copies**. The submitted model trains on the clean released labels.
5. Keep extras optional and guarded: `qiskit_aer` and `qiskit_ibm_runtime` imports inside `try/except` or inside the functions that use them. Hardware stays disabled by default and needs `QISKIT_IBM_TOKEN` in the environment and explicit approval. Never write a token into a file.
6. Seeds 0, 1, 2 everywhere a stochastic component exists (the forest and any attack randomness). State that the quantum-kernel SVM and the classical SVM are deterministic.
7. Keep a log of how many configurations were tried and how much tuning effort was used, for both the quantum model and its controls, with comparable effort.
8. Do not claim a quantum advantage, a robustness advantage or any speed claim. See section 11.

**Suggested layout**

```
challenge/submission.py      # Submission: quantum-kernel SVM + classical SVM + forest, vote, flag
                             # keep the interface: fit(X, y, seed=0), predict, predict_proba, resources()
challenge/guard.py           # attack lab (tamper, poison, missing) and alarm metrics
challenge/mitigation.py      # noisy simulation + calibration (optional qiskit_aer)
challenge/hardware_kernel.py # IBM inference for the kernel (optional, guarded, see section 7)
notebooks/CropGuard_Q.ipynb  # main run first, extras clearly separated and labeled "additional analysis"
outputs/                     # our predictions CSV (mimic results/reference_0_predictions.csv), tables, figures, hardware_result.json
```

**Working order**
1. Read `README.md`, `docs/EVALUATION.md`, `docs/HINTS.md`, `data/README.md`, the starter notebook and `challenge/*.py`. Run the starter notebook once in a fresh venv to confirm the environment.
2. Tier 1, then stop and report the numbers to the team before Tier 2.
3. Tier 2, then Tier 3 as extra analysis, then Tier 4 (hardware) following section 7, only once approval and credentials are in place.
4. Write plain-language comments for every non-obvious step. Teammates must be able to explain the notebook.

**Verification before saying "done"**
- Fresh venv with `python3.12 -m venv .venv` and `pip install -r requirements.txt`, then `jupyter nbconvert --to html --execute` on the notebook, with the runtime recorded.
- Confirm the main run needs no optional package and no credentials.
- Check the three seeds produce a mean and standard deviation table.
- Spot-check numbers against the saved outputs (no number in the slides that is not in an output file).
- Confirm no test labels or IDs are used as inputs.

---

## 9. Tested implementation notes

These snippets ran in a sandbox with the pinned versions. They are starting points to adapt, not final code.

### 9.1 Load the 8-number view

```python
import numpy as np
v = np.load("data/views/pca8.npz", allow_pickle=False)
Xtr, ytr = v["X_train"], v["y_train"]
Xva, yva = v["X_validation"], v["y_validation"]
Xte = v["X_test"]                      # test features only, no labels
labels = ["alfalfa", "maize", "wheat"]
# For integrity checks use challenge.data.load_data() (verifies checksums and splits).
```

### 9.2 Quantum kernel SVM (exact, Qiskit)

```python
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector
from sklearn.svm import SVC

N, SCALE = 8, 0.1                      # SCALE is "the knob": search about 0.05 to 0.25 on validation
fm = zz_feature_map(N, reps=1, entanglement="linear")

def states(X):
    return np.array([Statevector.from_instruction(fm.assign_parameters(x * SCALE)).data for x in X])

S_tr = states(Xtr)
K_tr = np.abs(S_tr @ S_tr.conj().T) ** 2          # fidelity kernel, train x train
K_va = np.abs(states(Xva) @ S_tr.conj().T) ** 2   # validation x train
svm_q = SVC(kernel="precomputed", C=10).fit(K_tr, ytr)
pred_q = svm_q.predict(K_va)
```

Resources to report: 8 qubits, circuit depth (`fm.decompose().depth()`), number of kernel entries or circuit evaluations (about N²/2 for training plus test-by-train for prediction), kernel storage (N × N floats), number of support vectors (`svm_q.n_support_`), and total runtime including the kernel construction.

**Optional speed-up.** For `reps=1`, linear entanglement, the state is `2^(-n/2) · exp(i·φ(b))` over bitstrings `b`, with `φ(b) = Σ_i 2·x_i·b_i + Σ_i 2·(π − x_i)(π − x_{i+1})·(b_i XOR b_{i+1})`. A vectorized NumPy version matched Qiskit in our tests. If you use it, validate against the Qiskit version on a few points and say so.

### 9.3 Classical comparison models

```python
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
svm_c = SVC(C=10, gamma="scale").fit(Xtr, ytr)
rf    = RandomForestClassifier(300, random_state=seed).fit(Xtr, ytr)
```

Choose `C`, `gamma` and the kernel scale from a small grid (no more than about 30 combinations in total) using validation, then freeze.

### 9.4 Vote and flag

```python
P = np.stack([svm_q.predict(K_X), svm_c.predict(X), rf.predict(X)])   # shape (3, n_fields)
unanimous = (P == P[0]).all(axis=0)
flag = ~unanimous                                    # send to human review
# majority vote; if all three differ, fall back to the classical SVM
final = np.where(P[0] == P[1], P[0], np.where(P[0] == P[2], P[0], P[1]))
```

Rows are `[quantum SVM, classical SVM, forest]`. If any two agree, that label wins. If all three differ, the classical SVM's label (row 1) is used.

### 9.5 Tampering attack (gradient-free random search)

```python
def attack(score_fn, X, y, eps, iters=150, seed=0):
    """score_fn(X) -> (n, 3) class scores. Finds a perturbation with L2 norm <= eps that lowers the true-class margin."""
    rng = np.random.default_rng(seed); d = np.zeros_like(X); ar = np.arange(len(y))
    def margin(D):
        s = score_fn(X + D); t = s[ar, y]; s2 = s.copy(); s2[ar, y] = -np.inf
        return t - s2.max(1)                       # > 0 correct, < 0 fooled
    best = margin(d)
    for it in range(iters):
        step = rng.normal(size=X.shape) * (eps * 0.5 if it < iters // 2 else eps * 0.15)
        cand = d + step
        cand *= np.minimum(1, eps / np.maximum(np.linalg.norm(cand, axis=1, keepdims=True), 1e-12))
        m = margin(cand); better = m < best
        d[better] = cand[better]; best = np.where(better, m, best)
    return X + d
```

`y` must be integer class indices. Use `SVC.decision_function` as the score for the SVMs and `predict_proba` for the forest. For the adaptive attacker, minimize the **largest** margin across the committee so that all models must be fooled. This is a weak attack, so say so.

- **Poisoning:** copy `ytr`, flip a fraction p of labels to another class at random, retrain, evaluate on clean validation. Average over several draws.
- **Missing features:** set k random PCA components per field to 0 (the training mean for PCA components) at evaluation time. Average over several draws.
- A more realistic tampering study should perturb the **13 raw bands** within plausible limits and pass them through the training-fitted preprocessing. Our quick tests perturbed the 8 compressed numbers.

### 9.6 Noise study and the error-fixer (optional extra, needs `qiskit-aer` for the simulator)

```python
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error, ReadoutError

def circ(x1, x2, fold, n=8):
    """Compute-uncompute circuit: ZZ map of x1 followed by the inverse ZZ map of x2. fold=3 triples each CX (ideal result unchanged, more noise)."""
    qc = QuantumCircuit(n)
    cx = lambda i: [qc.cx(i, i + 1) for _ in range(fold)]
    qc.h(range(n))
    for q in range(n): qc.p(2 * x1[q], q)
    for i in range(n - 1):
        cx(i); qc.p(2 * (np.pi - x1[i]) * (np.pi - x1[i + 1]), i + 1); cx(i)
    for i in reversed(range(n - 1)):
        cx(i); qc.p(-2 * (np.pi - x2[i]) * (np.pi - x2[i + 1]), i + 1); cx(i)
    for q in range(n): qc.p(-2 * x2[q], q)
    qc.h(range(n)); qc.measure_all()
    return qc          # kernel value = fraction of shots with the all-zeros bitstring

nm = NoiseModel()
r1 = thermal_relaxation_error(100e3, 80e3, 50); r2 = thermal_relaxation_error(100e3, 80e3, 400)
nm.add_all_qubit_quantum_error(depolarizing_error(0.002, 1).compose(r1), ["h"])
nm.add_all_qubit_quantum_error(depolarizing_error(0.015, 2).compose(r2.tensor(r2)), ["cx"])
nm.add_all_qubit_readout_error(ReadoutError([[0.97, 0.03], [0.04, 0.96]]))
sim = AerSimulator(method="density_matrix", noise_model=nm)
# counts = sim.run(circuits, shots=1024).result().get_counts(k)
```

Pass the features already multiplied by `SCALE` into `circ`. The function `circ` itself needs only Qiskit, so it is also the circuit used for the hardware run in section 7. Then:

1. Pick about 300 random training pairs. Their exact kernel value comes from the exact simulation in 9.2.
2. Run the same pairs on the noisy simulator (and, for the bonus, on hardware) to get noisy values.
3. Fit `LinearRegression` from noisy value to exact value. Apply it to all other pairs. Clip to [0, 1].
4. Report mean absolute error versus exact for raw, zero-noise extrapolation and the correction, at several noise multipliers, on pairs not used for calibration.

**Do not** simulate noise for all 914 training fields (about 418,000 pairs). Use a subset of about 60 training and 45 validation fields, as in section 5.4.

### 9.7 Paired bootstrap

Resample validation **spatial blocks** (from `data/validation_groups.json`) with replacement, recompute macro-F1 for both models on the same resample, and report the distribution of the difference. State the number of resamples and the seed.

---

## 10. Reporting checklist and slides

### Notebook must contain
- [ ] Short execution instructions at the top
- [ ] Selected model (quantum SVM plus the vote), matched classical-only ensemble, supplied reference, configurations
- [ ] Seeds 0, 1, 2: mean and standard deviation of macro-F1 and accuracy
- [ ] Per-class precision and recall, confusion matrices
- [ ] Training and prediction time, support-vector counts, kernel storage, qubits, depth, circuit evaluations
- [ ] Number of configurations tried and tuning effort, for the quantum model and its controls
- [ ] Paired bootstrap for any claimed difference
- [ ] Attack lab, alarm and noise study clearly labeled **additional analysis**
- [ ] **IBM hardware section** (backend, job ID, simulator comparison), off by default and reading the saved result
- [ ] Citations for reused code and methods

### Slides (3 to 5)
1. **Problem and question:** why crop checks need trust, and the one headline question
2. **System:** quantum inspector, classical inspector(s), vote and alarm, error-fixer (one diagram)
3. **Comparison:** quantum versus matched classical ensemble versus the reference (macro-F1 table, resources)
4. **Attacks and the alarm:** how badly tampering hurts, how much the alarm helps, the adaptive-attacker limit
5. **Findings and limitations:** what we learned (including that the quantum model is not more robust), the error-fixer result, and the **IBM hardware result** (backend, job ID, comparison) if done

---

## 11. Risks, limits and what must NOT be claimed

**Do not claim:**
- a quantum advantage of any kind (accuracy, robustness, speed or scaling)
- that the quantum model is more robust than the classical one (our tests say it is not)
- that the AI error-fixer improves final accuracy (only that it improves similarity values, unless a full run shows more)
- that the hardware demo measures accuracy (it is one small fixed-model inference example)
- anything about larger datasets, other regions or real devices beyond what was tested
- that the alarm stops a determined attacker who knows the defense (it did not)

**Known limits**
- Quick tests were tuned on 150 validation fields, so the real numbers will differ
- Attacks were simple, on the 8 compressed numbers rather than the 13 raw bands
- The noise study used a simulated device and a small subset. Real devices drift.
- The error-fixer needs exact simulation to produce calibration labels, which works only for small circuits
- The quantum kernel is very sensitive to the rotation-strength knob
- If the work is larger than expected, drop Tier 3 first, then keep Tiers 1, 2 and the hardware bonus, and keep the one-question story

**Questions to confirm with the organizers**
1. The submission route
2. Whether an IBM account and backend need approval, and whether optional extras (`qiskit-aer`) are acceptable in the submitted package
3. Judging-machine details for the runtime budget

---

## 12. Sources

- Challenge repository: <https://github.com/walid-mk/QiskitFallFest_EMSI_QuantumAfrica> (README, `docs/EVALUATION.md`, `docs/HINTS.md`, `data/README.md`)
- Ouassanouan et al. (2026), *Crop and irrigation types ground-truth dataset for Moroccan agricultural regions*, Scientific Data: <https://www.nature.com/articles/s41597-026-06993-y>
- NASA Landsat, *Saving millions in government dollars: Landsat helps fight crop insurance fraud*: <https://landsat.gsfc.nasa.gov/article/saving-millions-in-government-dollars-landsat-helps-fight-crop-insurance-fraud>
- Havlíček et al. (2019), *Supervised learning with quantum-enhanced feature spaces*: <https://arxiv.org/abs/1804.11326>
- Grossi et al. (2022), *Mixed Quantum–Classical Method for Fraud Detection With Quantum Feature Selection*: <https://arxiv.org/abs/2208.07963>
- Bowles, Ahmed and Schuld (2024), *Better than classical? The subtle art of benchmarking quantum machine learning models*: <https://arxiv.org/abs/2403.07059>
- Czaja et al. (2018), *Adversarial Examples in Remote Sensing*: <https://arxiv.org/abs/1805.10997>
- Dowling et al. (2024), *Adversarial Robustness Guarantees for Quantum Classifiers*: <https://arxiv.org/abs/2405.10360>
- Nowmi et al. (2025), *SoK: Critical Evaluation of Quantum Machine Learning for Adversarial Robustness*: <https://arxiv.org/abs/2511.14989>
- Onim and Thapliyal (2025), *Detection of Physiological Data Tampering Attacks with Quantum Machine Learning*: <https://arxiv.org/abs/2502.05966>
- Czarnik et al. (2020), *Error mitigation with Clifford quantum-circuit data*: <https://arxiv.org/abs/2005.10189>
- Liao et al. (2023), *Machine Learning for Practical Quantum Error Mitigation*: <https://arxiv.org/abs/2309.17368>
- Wang et al. (2021), *QuantumNAT: Quantum Noise-Aware Training*: <https://arxiv.org/abs/2110.11331>
- Shastry et al. (2022), *Shot-frugal and Robust quantum kernel classifiers*: <https://arxiv.org/abs/2210.06971>
- Du et al. (2020), *Quantum noise protects quantum classifiers against adversaries*: <https://arxiv.org/abs/2003.09416>

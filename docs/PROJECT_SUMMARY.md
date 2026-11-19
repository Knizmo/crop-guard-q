# CropGuard-Q: project summary

*A quantum crop checker that resists tampering and quantum noise.* Qiskit Fall Fest, EMSI × Quantum Africa, Moroccan Crop Classification Challenge. Written 19 November 2026.

**The idea in one minute.** Satellites help check which crop a farmer really planted, for example to catch false insurance or subsidy claims. If someone alters a field's satellite measurements, the model can be fooled. And quantum computers add errors of their own. CropGuard-Q tackles both: a quantum model that names the crop, an AI "noise-canceller" that cleans the quantum results, and a second, classical model that raises an alarm when the two disagree. Fields that trigger the alarm are sent for human inspection.

## 1. The problem

The challenge asks us to classify fields as alfalfa, maize or wheat from 13 satellite measurements per field. The data covers the Doukkala region: 914 training fields, 150 validation fields and 221 unlabeled test fields. The organizers' starter quantum model scores slightly below their classical baseline, and we are free to improve it or replace it with another quantum or hybrid model.

We add a security angle to the same task. Satellite crop checks are used in practice: NASA reports that Landsat imagery is used when investigators suspect fraudulent crop-insurance claims. A checker like this is only useful if it can be trusted, so we ask: what happens when the data is tampered with, and can we detect it?

## 2. Our solution in three parts

| Part | What it does | Everyday picture |
| --- | --- | --- |
| 1. Quantum inspector | An 8-qubit quantum circuit measures how similar two fields are (a "quantum kernel"). A standard classifier, an SVM, uses these similarities to name the crop. | A detective who compares each new field with fields already known. |
| 2. AI noise-canceller | Real quantum machines make errors. A small model learns to map noisy results to ideal ones, using about 300 pairs of fields whose exact answer we can compute by classical simulation. | Noise-cancelling headphones for the quantum computer. |
| 3. Tamper alarm | A quantum model and a classical model each classify the field. If they agree, the field is "verified". If they disagree, it is flagged for human review. | A second inspector who double-checks the first. |

How the pieces fit together:

1. Satellite measurements of a field come in and are compressed to 8 numbers (the challenge supplies this 8-number version).
2. The quantum inspector turns them into similarities with known fields. The AI noise-canceller cleans those similarities.
3. The classical inspector classifies the same field independently.
4. If both name the same crop, the field is verified. If not, it goes to a human.

## 3. What we have tested so far

These are quick feasibility tests, run in simulation. Robustness tests used the full 150-field validation set. The noise tests used a smaller subset (60 training and 45 validation fields) with simulated hardware noise. Treat all numbers as indicative: settings were chosen on the validation set, and 150 fields means roughly ±3 percentage points of uncertainty. The final report must use all 3 seeds and the test set.

### 3.1 Accuracy on clean data

| Model | Macro-F1 | Comment |
| --- | ---: | --- |
| Classical SVM on the 8-number view | 0.93 | Our classical control |
| Quantum kernel + SVM (8 qubits) | 0.91 | Our quantum inspector |
| Random forest (8-number view) | 0.89 | Alternative classical partner |
| Official reference (SVM, all 13 features) | 0.88 | Organizers' baseline |
| Official starter quantum model (4 qubits) | 0.86 | Organizers' starter |

Macro-F1 is the challenge's main metric (1.0 is perfect). The starter model uses only 4 of the 8 allowed qubits, which leaves clear room to improve.

**A key knob.** The quantum kernel is very sensitive to how strongly the data rotates the qubits. At the default strength, the score collapses to roughly 0.2–0.4. At a small strength (0.05–0.1), it reaches about 0.89–0.91. This is a clean, explainable finding for our slides.

### 3.2 Attacks: tampering, poisoning and missing data

| Threat | Setting | Classical | Quantum |
| --- | --- | ---: | ---: |
| Tampering with a field's measurements | Small change | 0.80 | 0.79 |
| | Medium change | 0.59 | 0.60 |
| | Large change | 0.23 | 0.22 |
| Poisoned training labels | 10% wrong labels | 0.91 | 0.88 |
| | 20% wrong labels | 0.88 | 0.86 |
| | 30% wrong labels | 0.84 | 0.83 |
| Missing measurements (sensor dropout) | 2 of 8 missing | 0.79 | 0.79 |
| | 4 of 8 missing | 0.62 | 0.64 |

Tampering rows show accuracy after an attack aimed directly at that model. Poisoning and missing-data rows show macro-F1. Tampering used a simple random-search attack with a fixed size limit.

**Main finding:** the quantum model is not more robust than the classical one in these tests. Both are about equally fragile. We will report this honestly instead of claiming a quantum advantage.

### 3.3 The tamper alarm

| Setup | Attack gets through | Clean fields flagged | Accuracy of verified fields |
| --- | ---: | ---: | ---: |
| Main classical model alone | 41% | 0% | 93% |
| + quantum partner | 28% | 6% | 95% |
| + classical partner (forest) | 23% | 7% | 94% |
| + both partners | 20% | 11% | 96% |

"Attack gets through" means the attacker fooled all models into the same wrong crop, using a medium-sized change aimed at the main model only. The last column is measured on clean fields.

- The alarm roughly halves the number of attacks that succeed silently, at the cost of flagging about 6–11% of clean fields for review.
- A quantum partner works about as well as a classical partner. The quantum part does not make the alarm uniquely better.
- An attacker who knows the whole committee still gets through 27–31% of the time. We will state this limit clearly.

### 3.4 The AI noise-canceller

We simulated a noisy device (gate errors, qubit decay, readout errors, 1024 shots per circuit) and measured how far the noisy similarity values were from the exact ones. Lower is better.

| Method | Error, normal noise | Error, 3× noise | Circuit runs needed |
| --- | ---: | ---: | ---: |
| Raw noisy result | 0.081 | 0.137 | 1× |
| Standard zero-noise extrapolation | 0.059 | 0.128 | 2× |
| AI correction (simple linear fit) | 0.013 | 0.037 | 1× |

Errors are measured on pairs of fields the AI never saw during calibration (about 300 calibration pairs were used).

- The AI correction is about 6× more accurate than the raw result and about 4× more accurate than the standard method, with fewer circuit runs.
- A simple straight-line fit did as well as boosted trees, so heavy AI is not needed. Published work also describes noise as roughly a scale-and-shift of the true result.
- **Not yet proven:** better similarity values did not clearly translate into better final accuracy. The classifier already tolerated this noise level, and our 45-field test is too small to separate the methods. We will claim "more accurate similarities" until a full run shows more.

## 4. How this fits the judging

| Criterion | Points | How we address it |
| --- | ---: | --- |
| Predictive performance | 30 | 30 × test macro-F1. Validation results of about 0.91–0.93 are above the 0.88 reference, to be confirmed on the test set. |
| Experimental quality and analysis | 40 | Fair classical controls, three threat types, an alarm tested against an attacker who knows it, 3 seeds, runtime and resource reporting, and honest negative findings. The rules give full credit to well-supported negative results. |
| Model design and reproducibility | 30 | Justified encoding, fit on training data only, fixed seeds, one runnable notebook with clear instructions. |
| Bonus | +5 | Run the quantum part on real IBM hardware and compare against the ideal simulator, using the AI correction. Optional. |

**Rules we follow:** at most 8 qubits with exact local simulation for the main results; transformations fitted on training data only; validation used for choosing settings and the test set only for final predictions; seeds 0, 1 and 2; and reporting of macro-F1, accuracy, per-class precision and recall, confusion matrices, runtime and model resources.

**Fair controls:** the classical SVM on the same 8 numbers, a classical alternative partner (random forest), and, planned, a quantum kernel without entanglement to test whether entanglement matters. The starter's own classical control scores only 0.58 because it is very small, so it is too weak to be a fair comparison and we avoid relying on it.

## 5. Plan until the deadline

The deadline is Saturday evening. The plan assumes about 6–8 hours of work per day and leaves a buffer.

| When | What |
| --- | --- |
| Thursday | Set up and run the starter notebook. Build the quantum kernel, the classical controls and 3 seeds. At the end of the day, something submittable exists. |
| Friday morning | Attack lab (tampering, poisoning, missing data) and the tamper alarm. |
| Friday afternoon | AI noise-canceller on a noisy simulator. Draft the slides in the evening. |
| Saturday morning | Final full runs, results tables, final slides (3–5). |
| Saturday afternoon | Submit with a buffer. If the IBM run is not finished by about 2 pm, drop it and use the simulator results. |

**To do today:**

- [ ] Request an IBM Quantum account and token.
- [ ] Ask the organizers for the submission route and the deadline's timezone, which the starter repository still lists as to be announced.
- [ ] Split the work between teammates.

If time runs short, cut in this order: the hardware run first, then the AI noise-canceller. The attack lab and the alarm stay, because they carry most of the originality.

## 6. Risks and limits

- Noise results come from a simulated device. Real devices drift, so the correction must be recalibrated for each session.
- The AI correction learns from exact simulation of calibration pairs. That works because our circuits are small (8 qubits or fewer).
- Our attacks were simple and applied to the 8 compressed features. The real project should perturb the 13 raw bands and test stronger attacks.
- Numbers above were tuned on the validation set and are indicative. The final claims must come from the full 3-seed runs.
- The quantum model is not more robust than the classical one in our tests, and we do not claim a quantum advantage.

## 7. Glossary

| Term | Meaning |
| --- | --- |
| Qubit | The quantum version of a bit. We use up to 8. |
| Quantum kernel | A way to use a quantum circuit to score how similar two data points are. |
| SVM | A standard classifier that can use such similarity scores. |
| Macro-F1 | An accuracy score averaged fairly over the three crops. 1.0 is perfect. |
| PCA | A method that compresses many measurements into a few numbers. The challenge supplies 4- and 8-number versions. |
| Tampering (adversarial attack) | Small deliberate changes to the input meant to make the model answer wrongly. |
| Poisoning | Planting wrong labels in the training data to damage the model. |
| Error mitigation | Reducing the effect of hardware noise on results without full error correction. |
| Zero-noise extrapolation | A standard mitigation trick: run at higher noise levels and extrapolate back to zero noise. |

## 8. Sources

- Challenge repository, Qiskit Fall Fest EMSI × Quantum Africa
- Ouassanouan et al. (2026), *Crop and irrigation types ground-truth dataset for Moroccan agricultural regions*, Scientific Data
- NASA Landsat: *Saving millions in government dollars, Landsat helps fight crop insurance fraud*
- Czaja et al. (2018), *Adversarial Examples in Remote Sensing*
- Dowling et al. (2024), *Adversarial Robustness Guarantees for Quantum Classifiers*
- Nowmi et al. (2025), *SoK: Critical Evaluation of Quantum Machine Learning for Adversarial Robustness*
- Onim and Thapliyal (2025), *Detection of Physiological Data Tampering Attacks with Quantum Machine Learning*
- Czarnik et al. (2020), *Error mitigation with Clifford quantum-circuit data*
- Liao et al. (2023), *Machine Learning for Practical Quantum Error Mitigation* (IBM Quantum)
- Wang et al. (2021), *QuantumNAT: Quantum Noise-Aware Training*
- Shastry et al. (2022), *Shot-frugal and Robust quantum kernel classifiers*
- Du et al. (2020), *Quantum noise protects quantum classifiers against adversaries*
- Google DeepMind, *AlphaQubit tackles one of quantum computing's biggest challenges*

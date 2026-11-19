"""CropGuard-Q demo: quantum inspector, tamper alarm and AI noise-canceller, end to end in ~15 s.

Run:  .venv/bin/python -m pip install -r requirements-noise.txt   (once)
      .venv/bin/python demo.py
Settings were picked on the validation set, so every number here is indicative, not a final result.
"""
import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, thermal_relaxation_error
from sklearn.svm import SVC
from challenge.data import ROOT, load_data
from challenge.kernel import QuantumKernelSVM, fidelity
from challenge.metrics import score

EPS = 0.35  # tamper budget per feature (features are standardized); about the summary's "medium" change
rng = np.random.default_rng(0)


def tamper(predict, X, y, eps, tries=200):
    """Random-search attack: try random changes within +/-eps per feature until the model is wrong."""
    X_adv = X.copy()
    for i in range(len(X)):
        if predict(X[i:i + 1])[0] != y[i]:
            continue  # already misclassified, nothing to attack
        candidates = X[i] + rng.uniform(-eps, eps, (tries, X.shape[1]))
        hit = np.flatnonzero(predict(candidates) != y[i])
        if hit.size:
            X_adv[i] = candidates[hit[0]]
    return X_adv


def noise_model(level):
    """Simulated device: gate errors, qubit decay (T1/T2, in ns) and readout errors; level 3 = 3x worse."""
    nm = NoiseModel()
    t1, t2 = 150e3 / level, 100e3 / level
    nm.add_all_qubit_quantum_error(depolarizing_error(5e-4 * level, 1).compose(thermal_relaxation_error(t1, t2, 35)), ['sx', 'x'])
    decay = thermal_relaxation_error(t1, t2, 300)
    nm.add_all_qubit_quantum_error(depolarizing_error(8e-3 * level, 2).compose(decay.tensor(decay)), ['cx'])
    r = 0.015 * level
    nm.add_all_qubit_readout_error(ReadoutError([[1 - r, r], [r, 1 - r]]))
    return nm


print('=== 0. Data ===')
load_data()  # verifies the checksum of every supplied file, including views/pca8.npz
with np.load(ROOT / 'data/views/pca8.npz', allow_pickle=False) as f:
    v = {k: f[k] for k in f.files}
Xtr, ytr, Xva, yva, labels = v['X_train'], v['y_train'], v['X_validation'], v['y_validation'], v['labels'].tolist()
print(f'{len(Xtr)} training and {len(Xva)} validation fields, each compressed to 8 numbers; crops: {labels}')

print('\n=== 1. Quantum inspector: 8-qubit kernel + SVM ===')
print('Each field becomes an 8-qubit quantum state; the SVM compares fields by how much their states overlap.')
quantum = QuantumKernelSVM(scale=0.1, C=10).fit(Xtr, ytr)
classical = SVC(C=10).fit(Xtr, ytr)
for name, model in [('Quantum kernel + SVM', quantum), ('Classical RBF SVM, same 8 numbers', classical)]:
    s = score(yva, model.predict(Xva), labels)
    print(f'  {name:34s} macro-F1 {s["macro_f1"]:.3f}   accuracy {s["accuracy"]:.1%}')
collapsed = score(yva, QuantumKernelSVM(scale=1.0, C=10).fit(Xtr, ytr).predict(Xva), labels)['macro_f1']
print(f'  Key knob: same kernel at the default rotation strength 1.0 -> macro-F1 {collapsed:.3f}'
      ' (every field looks unlike every other)')
print(f'  Resources: {quantum.resources()}')

print('\n=== 2. Tamper alarm: quantum and classical inspectors must agree ===')
qc, cc = quantum.predict(Xva), classical.predict(Xva)
agree = qc == cc
print(f'  Clean fields: {agree.mean():.0%} verified, {1 - agree.mean():.0%} flagged for review;'
      f' accuracy of verified fields {np.mean(cc[agree] == yva[agree]):.0%}')
X_adv = tamper(classical.predict, Xva, yva, EPS)
assert np.abs(X_adv - Xva).max() <= EPS, 'attack exceeded its budget'
ca, qa = classical.predict(X_adv), quantum.predict(X_adv)
print(f'  Attacker changes each measurement by up to +/-{EPS} (std units), aiming at the classical model:')
print(f'    classical alone:      wrong on {np.mean(ca != yva):.0%} of fields, nobody notices')
print(f'    with quantum partner: {np.mean((ca != yva) & (qa == ca)):.0%} get through silently,'
      f' {np.mean(qa != ca):.0%} flagged for a human')
for i in np.flatnonzero((ca != yva) & (qa != ca))[:1]:
        print(f'  Example field {v["ids_validation"][i]}: really {yva[i]}; after tampering the classical model says'
          f' {ca[i]}, the quantum model says {qa[i]} -> FLAGGED')

print('\n=== 3. AI noise-canceller: fixing similarities measured on a noisy quantum device ===')
pairs = rng.choice(len(Xtr), (400, 2))  # training fields only: 300 pairs to calibrate, 100 held out
exact = fidelity(quantum.states(Xtr[pairs[:, 0]]), quantum.states(Xtr[pairs[:, 1]])).diagonal()
circuits = [quantum.pair_circuit(Xtr[a], Xtr[b]) for a, b in pairs]
assert np.isclose(Statevector(circuits[0].remove_final_measurements(inplace=False)).probabilities()[0], exact[0])
# num_processes=1: parallel transpile pickles with dill, which breaks on Python 3.15
circuits = transpile(circuits, basis_gates=['rz', 'sx', 'x', 'cx'], seed_transpiler=0, num_processes=1)
print(f'  One similarity = one 8-qubit circuit (depth {circuits[0].depth()}, {circuits[0].count_ops()["cx"]} CX gates),'
      ' 1024 shots, score = share of all-zero outcomes')
for level in (1, 3):
    counts = AerSimulator(noise_model=noise_model(level), seed_simulator=0).run(circuits, shots=1024).result().get_counts()
    noisy = np.array([c.get('0' * 8, 0) / 1024 for c in counts])
    a, b = np.polyfit(noisy[:300], exact[:300], 1)  # the noise-canceller: a learned straight-line correction
    raw, fixed = np.abs(noisy[300:] - exact[300:]).mean(), np.abs(a * noisy[300:] + b - exact[300:]).mean()
    print(f'  {level}x noise: average error raw {raw:.3f} -> corrected {fixed:.3f}'
          f' ({raw / fixed:.0f}x better) on 100 pairs never used for calibration')

print('\nNext: run these same circuits on a real IBM device and compare with the exact values (plan, Phase 4).')

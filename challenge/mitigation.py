"""Noise study and the AI error-fixer (additional analysis; the simulator needs requirements-noise.txt).

A real device returns noisy similarity values. Because 8 qubits are cheap to simulate exactly, we know the
true value for any training pair, so a linear regression can learn noisy -> exact on calibration pairs and
correct pairs it never saw. Zero-noise extrapolation (ZNE: run again with every CX tripled, then extrapolate
to zero noise) is the classical baseline that costs twice the circuits.
"""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC

from .kernel import N, fidelity
from .metrics import score
from .submission import LABELS

BASIS = ['rz', 'sx', 'x', 'cx']


def circuit(x1, x2, fold=1):
    """Compute-uncompute ZZ circuit on already-scaled features: P(all zeros) = |<phi(x2)|phi(x1)>|^2.
    fold=3 triples each CX: same ideal result, about three times the two-qubit noise."""
    qc = QuantumCircuit(N)
    cx = lambda i: [qc.cx(i, i + 1) for _ in range(fold)]
    qc.h(range(N))
    for q in range(N): qc.p(2 * x1[q], q)
    for i in range(N - 1):
        cx(i); qc.p(2 * (np.pi - x1[i]) * (np.pi - x1[i + 1]), i + 1); cx(i)
    for i in reversed(range(N - 1)):
        cx(i); qc.p(-2 * (np.pi - x2[i]) * (np.pi - x2[i + 1]), i + 1); cx(i)
    for q in range(N): qc.p(-2 * x2[q], q)
    qc.h(range(N))
    qc.measure_all()
    return qc


def noise_model(level):
    """Simulated device: gate errors, qubit decay (T1/T2 in ns) and readout errors; level 3 = 3x worse."""
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, thermal_relaxation_error
    nm = NoiseModel()
    t1, t2 = 150e3 / level, 100e3 / level
    nm.add_all_qubit_quantum_error(depolarizing_error(5e-4 * level, 1).compose(thermal_relaxation_error(t1, t2, 35)), ['sx', 'x'])
    decay = thermal_relaxation_error(t1, t2, 300)
    nm.add_all_qubit_quantum_error(depolarizing_error(8e-3 * level, 2).compose(decay.tensor(decay)), ['cx'])
    r = 0.015 * level
    nm.add_all_qubit_readout_error(ReadoutError([[1 - r, r], [r, 1 - r]]))
    return nm


def ideal_zeros(qc):
    """Exact P(all zeros) of a measured circuit."""
    return Statevector(qc.remove_final_measurements(inplace=False)).probabilities()[0]


def zeros_probability(counts, shots):
    return np.array([c.get('0' * N, 0) / shots for c in counts])


def fit_fixer(noisy, exact):
    """The AI error-fixer: one straight line, noisy value -> exact value."""
    return LinearRegression().fit(np.asarray(noisy)[:, None], exact)


def apply_fixer(fixer, noisy):
    return np.clip(fixer.predict(np.asarray(noisy)[:, None]), 0, 1)


def to_basis(circuits):
    # optimization_level=0 keeps folded CX triples; num_processes=1 avoids dill pickling issues on new Pythons
    return transpile(circuits, basis_gates=BASIS, optimization_level=0, seed_transpiler=0, num_processes=1)


def run_noisy(isa, level, shots, seed):
    from qiskit_aer import AerSimulator
    sim = AerSimulator(noise_model=noise_model(level), seed_simulator=seed, method='density_matrix', max_parallel_experiments=0)
    counts = sim.run(isa, shots=shots).result().get_counts()
    return zeros_probability(counts if isinstance(counts, list) else [counts], shots)


def noise_study(quantum, Ztr, ytr, Zva, yva, levels=(1, 2, 3), n_train=40, n_val=30, n_cal=200, shots=1024, seed=0):
    """Exact vs noisy vs corrected similarity values and the SVM trained on each, on a small subset."""
    rng = np.random.default_rng(seed)
    tr = rng.choice(len(Ztr), n_train, replace=False)
    va = rng.choice(len(Zva), n_val, replace=False)
    rest = np.setdiff1d(np.arange(len(Ztr)), tr)
    cal = rng.choice(rest, (n_cal, 2))  # calibration pairs: training fields outside the subset
    s = quantum.scale
    S = quantum.states
    exact_cal = fidelity(S(Ztr[cal[:, 0]]), S(Ztr[cal[:, 1]])).diagonal()
    K_tr, K_va = fidelity(S(Ztr[tr]), S(Ztr[tr])), fidelity(S(Zva[va]), S(Ztr[tr]))
    iu = np.triu_indices(n_train, 1)
    pairs = ([(Ztr[a], Ztr[b]) for a, b in cal] + [(Ztr[tr[i]], Ztr[tr[j]]) for i, j in zip(*iu)]
             + [(Zva[i], Ztr[j]) for i in va for j in tr])
    circuits = [circuit(s * a, s * b) for a, b in pairs]
    assert np.isclose(ideal_zeros(circuits[0]), exact_cal[0], atol=1e-9), 'circuit and exact kernel disagree'
    isa, isa3 = to_basis(circuits), to_basis([circuit(s * Zva[i], s * Ztr[j], fold=3) for i in va for j in tr])
    svm = lambda Ktr, Kva: SVC(kernel='precomputed', C=quantum.svc.C).fit(Ktr, ytr[tr]).predict(Kva)
    f1 = lambda K1, K2: score(yva[va], svm(K1, K2), list(LABELS))['macro_f1']
    rows = []
    for level in levels:
        p, p3 = run_noisy(isa, level, shots, seed), run_noisy(isa3, level, shots, seed)
        cal_p, tr_p, va_p = p[:n_cal], p[n_cal:n_cal + len(iu[0])], p[n_cal + len(iu[0]):]
        fixer = fit_fixer(cal_p, exact_cal)
        noisy_tr = np.eye(n_train); noisy_tr[iu] = tr_p; noisy_tr = np.maximum(noisy_tr, noisy_tr.T)
        fixed_tr = np.eye(n_train); fixed_tr[iu] = apply_fixer(fixer, tr_p); fixed_tr = np.maximum(fixed_tr, fixed_tr.T)
        va_noisy, va_fixed = va_p.reshape(n_val, n_train), apply_fixer(fixer, va_p).reshape(n_val, n_train)
        zne = np.clip(1.5 * va_p - 0.5 * p3, 0, 1).reshape(n_val, n_train)  # linear extrapolation from folds 1 and 3
        mae = lambda K: float(np.abs(K - K_va).mean())
        rows.append(dict(level=level, shots=shots, mae_raw=mae(va_noisy), mae_zne=mae(zne), mae_fixed=mae(va_fixed),
                         fixer_slope=float(fixer.coef_[0]), fixer_intercept=float(fixer.intercept_),
                         f1_exact=f1(K_tr, K_va), f1_raw=f1(noisy_tr, va_noisy), f1_fixed=f1(fixed_tr, va_fixed),
                         circuits=len(isa) + len(isa3), depth=isa[0].depth(), cx=isa[0].count_ops().get('cx', 0),
                         sample_exact=K_va.ravel()[:400].tolist(), sample_raw=va_noisy.ravel()[:400].tolist(),
                         sample_fixed=va_fixed.ravel()[:400].tolist()))
    return dict(rows=rows, n_train=n_train, n_val=n_val, n_cal=n_cal, evaluation_pairs=n_val * n_train)

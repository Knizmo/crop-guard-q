"""Noise study and the AI error-fixer (additional analysis; the simulator needs requirements-noise.txt).

A real device returns noisy similarity values. Because 8 qubits are cheap to simulate exactly, we know the
true value for any training pair, so a linear regression can learn noisy -> exact on calibration pairs and
correct pairs it never saw. Zero-noise extrapolation (ZNE: run again with every CX tripled, then extrapolate
to zero noise) is the classical baseline that costs twice the circuits.

The simulated device is built from REAL calibration data of an IBM QPU (configs/experiment.json "noise.device"):
median gate errors, gate durations, readout error and T1/T2, read live from the backend or from IBM's
offline snapshot of it. `level` multiplies those error rates to show how the methods degrade.
"""
import json
from datetime import datetime, timezone

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC

from .config import EXPERIMENT, OUT
from .kernel import fidelity
from .metrics import score
from .submission import LABELS

BASIS = ['rz', 'sx', 'x', 'cx']
NOISE = EXPERIMENT['noise']
PROFILE = OUT / 'device_profile.json'


def circuit(x1, x2, fold=1):
    """Compute-uncompute ZZ circuit on already-scaled features: P(all zeros) = |<phi(x2)|phi(x1)>|^2.
    fold=3 triples each CX: same ideal result, about three times the two-qubit noise."""
    n = len(x1)
    qc = QuantumCircuit(n)
    cx = lambda i: [qc.cx(i, i + 1) for _ in range(fold)]
    qc.h(range(n))
    for q in range(n): qc.p(2 * x1[q], q)
    for i in range(n - 1):
        cx(i); qc.p(2 * (np.pi - x1[i]) * (np.pi - x1[i + 1]), i + 1); cx(i)
    for i in reversed(range(n - 1)):
        cx(i); qc.p(-2 * (np.pi - x2[i]) * (np.pi - x2[i + 1]), i + 1); cx(i)
    for q in range(n): qc.p(-2 * x2[q], q)
    qc.h(range(n))
    qc.measure_all()
    return qc


def device_profile(backend):
    """Median calibration of a real IBM device: a live backend (hardware_kernel.connect) or a fake_provider snapshot."""
    t = backend.target
    two = next(g for g in ('cz', 'ecr', 'cx') if g in t.operation_names)
    med = lambda values: float(np.median([v for v in values if v is not None]))
    gate = lambda name, attr: med(getattr(p, attr) for p in t[name].values() if p is not None)
    return dict(device=backend.name, retrieved_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                qubits=backend.num_qubits, two_qubit_gate=two,
                sx_error=gate('sx', 'error'), sx_seconds=gate('sx', 'duration'),
                two_qubit_error=gate(two, 'error'), two_qubit_seconds=gate(two, 'duration'),
                readout_error=gate('measure', 'error'),
                t1_seconds=med(q.t1 for q in t.qubit_properties if q), t2_seconds=med(q.t2 for q in t.qubit_properties if q))


def save_profile(backend):
    """Record a live backend's calibration so offline runs reuse it (no token needed afterwards)."""
    profile = device_profile(backend)
    PROFILE.parent.mkdir(exist_ok=True)
    PROFILE.write_text(json.dumps(profile, indent=2))
    return profile


def load_profile(device=NOISE['device']):
    """Saved live calibration of `device` (outputs/device_profile.json), else IBM's offline snapshot of it."""
    if PROFILE.exists():
        profile = json.loads(PROFILE.read_text())
        if profile['device'] == device: return profile
    from qiskit_ibm_runtime import fake_provider  # requirements-hardware.txt
    return device_profile(getattr(fake_provider, 'Fake' + device.removeprefix('ibm_').capitalize())())


def noise_model(profile, level=1):
    """Uniform device from real calibration medians; `level` multiplies every error rate and divides T1/T2.
    shortcut: every qubit gets the median values (no crosstalk, drift or per-qubit spread), and the reported gate
    error and the relaxation are both applied, so the model is slightly pessimistic; per-qubit models need a layout."""
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, thermal_relaxation_error
    t1 = profile['t1_seconds'] / level
    t2 = min(profile['t2_seconds'], 2 * profile['t1_seconds']) / level  # physical limit T2 <= 2 T1
    relax = lambda seconds: thermal_relaxation_error(t1, t2, seconds)
    # average gate error e -> depolarizing parameter e * d / (d - 1), d = 2 (one qubit) or 4 (two qubits)
    one = depolarizing_error(min(2 * profile['sx_error'] * level, 1), 1).compose(relax(profile['sx_seconds']))
    decay = relax(profile['two_qubit_seconds'])
    two = depolarizing_error(min(4 / 3 * profile['two_qubit_error'] * level, 1), 2).compose(decay.tensor(decay))
    r = min(profile['readout_error'] * level, 0.5)
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(one, ['sx', 'x'])
    nm.add_all_qubit_quantum_error(two, ['cx'])
    nm.add_all_qubit_readout_error(ReadoutError([[1 - r, r], [r, 1 - r]]))
    return nm


def ideal_zeros(qc):
    """Exact P(all zeros) of a measured circuit."""
    return Statevector(qc.remove_final_measurements(inplace=False)).probabilities()[0]


def zeros_probability(counts, shots, n):
    return np.array([c.get('0' * n, 0) / shots for c in counts])


def fit_fixer(noisy, exact):
    """The AI error-fixer: one straight line, noisy value -> exact value."""
    return LinearRegression().fit(np.asarray(noisy)[:, None], exact)


def apply_fixer(fixer, noisy):
    return np.clip(fixer.predict(np.asarray(noisy)[:, None]), 0, 1)


def to_basis(circuits):
    # optimization_level=0 keeps folded CX triples; num_processes=1 avoids dill pickling issues on new Pythons
    return transpile(circuits, basis_gates=BASIS, optimization_level=0, seed_transpiler=0, num_processes=1)


def run_noisy(isa, profile, level, shots, seed):
    from qiskit_aer import AerSimulator
    sim = AerSimulator(noise_model=noise_model(profile, level), seed_simulator=seed, method='density_matrix',
                       max_parallel_experiments=0)
    counts = sim.run(isa, shots=shots).result().get_counts()
    return zeros_probability(counts if isinstance(counts, list) else [counts], shots, isa[0].num_clbits)


def noise_study(quantum, Ztr, ytr, Zva, yva, profile, levels=NOISE['levels'], n_train=NOISE['train_fields'],
                n_val=NOISE['validation_fields'], n_cal=NOISE['calibration_pairs'], shots=NOISE['shots'], seed=NOISE['seed']):
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
        p, p3 = run_noisy(isa, profile, level, shots, seed), run_noisy(isa3, profile, level, shots, seed)
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
    return dict(rows=rows, n_train=n_train, n_val=n_val, n_cal=n_cal, evaluation_pairs=n_val * n_train, profile=profile)

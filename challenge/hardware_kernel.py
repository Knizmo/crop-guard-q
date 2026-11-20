"""IBM hardware bonus: inference-only quantum-kernel values on a real QPU, compared with the exact simulator.

Opt-in only: needs requirements-hardware.txt, QISKIT_IBM_TOKEN in the environment (never in a file) and
approved=True. The committee is trained locally and frozen. The QPU only measures the similarity circuits
between a few validation fields and the quantum SVM's support vectors; nothing is trained on hardware.
Calibration pairs of training fields (exact value known) ride in the same job, so the AI error-fixer is
fitted on that device in that session. One small fixed-model inference example, not an accuracy estimate.
"""
import json
import os
from datetime import datetime, timezone

import numpy as np

from .data import ROOT
from .kernel import fidelity
from .mitigation import apply_fixer, circuit, fit_fixer
from .submission import LABELS

RESULT = ROOT / 'outputs/hardware_result.json'


def connect(backend_name=None):
    from qiskit_ibm_runtime import QiskitRuntimeService
    # Credentials stay in the user's environment, never in the notebook, outputs or git.
    service = QiskitRuntimeService(channel='ibm_quantum_platform', token=os.environ['QISKIT_IBM_TOKEN'])
    if backend_name: return service.backend(backend_name)
    return service.least_busy(operational=True, simulator=False, min_num_qubits=8)


def build(sub, X_fields, n_cal=40, seed=0):
    """Circuits: every field against every support vector, then n_cal training calibration pairs."""
    q = sub.members[0]; s = q.scale
    Z, Zsv = sub.transform(X_fields), q.train_X[q.svc.support_]
    cal = np.random.default_rng(seed).choice(len(q.train_X), (n_cal, 2))
    circuits = [circuit(s * z, s * x) for z in Z for x in Zsv]
    circuits += [circuit(s * q.train_X[a], s * q.train_X[b]) for a, b in cal]
    exact = np.r_[fidelity(q.states(Z), q.states(Zsv)).ravel(),
                  fidelity(q.states(q.train_X[cal[:, 0]]), q.states(q.train_X[cal[:, 1]])).diagonal()]
    return circuits, exact


def pick_fields(sub, X_train, Xv, yv, ids, eps=0.1, seed=0):
    """One typical verified field per crop, plus one tampered field that the alarm flags.
    'Typical' = median quantum margin among eligible fields, so the demo is neither the easiest nor the hardest case."""
    from .guard import index, margin, member_margin, tamper
    yi = index(yv)
    def typical(X, ok):
        m = margin(sub.members[0].decision_function(sub.transform(X)), yi)
        cand = np.flatnonzero(ok)
        return cand[np.argsort(m[cand])[len(cand) // 2]]
    r = sub.inspect(Xv)
    fields = [dict(name=f'clean {c}', sample_id=str(ids[i]), target=c, X=Xv[i].tolist())
              for c in LABELS for i in [typical(Xv, ~r['flag'] & (r['prediction'] == yv) & (yv == c))]]
    adv = tamper(member_margin(sub, 1), Xv, yi, eps, X_train, seed=seed)
    a = sub.inspect(adv)
    i = typical(adv, a['flag'] & (a['votes'][1] != yv) & (a['votes'][0] == yv))  # RBF fooled, quantum still right
    fields.append(dict(name=f'tampered {yv[i]} (eps {eps})', sample_id=str(ids[i]), target=str(yv[i]), X=adv[i].tolist()))
    return fields


def quantum_votes(sub, values, n_fields):
    """SVM decision from measured kernel values; only support-vector columns enter the decision."""
    q = sub.members[0]
    K = np.zeros((n_fields, len(q.train_X)))
    K[:, q.svc.support_] = np.asarray(values).reshape(n_fields, -1)
    return q.svc.decision_function(K), q.svc.predict(K)


def residuals(hw):
    """Post-fixer errors of a saved hardware run on its field x support-vector pairs (calibration pairs excluded)."""
    exact, measured = np.array(hw['exact']), np.array(hw['measured'])
    n = hw['support_vectors'] * len(hw['fields'])
    return apply_fixer(fit_fixer(measured[n:], exact[n:]), measured[:n]) - exact[:n]


def robustness_scan(Z, y, Zv, yv, errors, scale, Cs=(0.3, 1, 3, 10, 30, 100), shots=512, draws=20, seed=0):
    """Offline, no QPU: how often does each C keep its exact validation vote under 512-shot sampling noise,
    and under kernel errors resampled from a real hardware run? Large C means large dual coefficients,
    which amplify small kernel errors."""
    from .kernel import QuantumKernelSVM
    from .metrics import score
    rng, rows = np.random.default_rng(seed), []
    for C in Cs:
        q = QuantumKernelSVM(scale=scale, C=C).fit(Z, y)
        K = q.kernel(Zv); exact = q.svc.predict(K)
        agree = lambda noisy: np.mean([(q.svc.predict(noisy()) == exact).mean() for _ in range(draws)])
        rows.append(dict(C=C, support_vectors=int(q.svc.n_support_.sum()), max_dual=float(np.abs(q.svc.dual_coef_).max()),
                         exact_macro_f1=score(yv, exact, list(LABELS))['macro_f1'],
                         agree_shot_noise=agree(lambda: rng.binomial(shots, np.clip(K, 0, 1)) / shots),
                         agree_hardware_error=agree(lambda: np.clip(K + rng.choice(errors, K.shape), 0, 1))))
    return rows


def submit(circuits, backend, shots=512, approved=False):
    if not approved: raise PermissionError('Obtain approval for this hardware job first')
    if not 1 <= shots <= 4096: raise ValueError('Use 1-4096 inference shots')
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2
    isa = generate_preset_pass_manager(backend=backend, optimization_level=2, seed_transpiler=42).run(circuits, num_processes=1)
    job = SamplerV2(mode=backend).run(isa, shots=shots)
    print('Inference-only job:', job.job_id(), 'backend:', backend.name, flush=True)
    return job, isa


def zeros(job, shots):
    return np.array([r.data.meas.get_counts().get('0' * 8, 0) / shots for r in job.result()])


def run(sub, fields, backend, shots=512, n_cal=40, approved=False, save=RESULT):
    """fields: list of dicts with name, sample_id, target and X (13 raw values). Returns and saves the comparison."""
    X = np.array([f['X'] for f in fields], dtype=float)
    circuits, exact = build(sub, X, n_cal)
    job, isa = submit(circuits, backend, shots, approved)
    measured = zeros(job, shots)
    n = len(fields) * len(sub.members[0].svc.support_)
    fixer = fit_fixer(measured[n:], exact[n:])
    fixed = apply_fixer(fixer, measured[:n])
    d_exact, v_exact = quantum_votes(sub, exact[:n], len(fields))
    d_raw, v_raw = quantum_votes(sub, measured[:n], len(fields))
    d_fixed, v_fixed = quantum_votes(sub, fixed, len(fields))
    local = sub.votes(X)
    ops = isa[0].count_ops()
    try: usage = job.usage()
    except Exception: usage = None
    out = dict(
        backend=backend.name, job_id=job.job_id(), shots=shots, circuits=len(circuits), qubits=8,
        support_vectors=int(len(sub.members[0].svc.support_)), calibration_pairs=n_cal,
        transpiled_depth=int(isa[0].depth()), transpiled_two_qubit_gates=int(sum(ops.get(g, 0) for g in ('cz', 'ecr', 'cx'))),
        physical_qubits=[int(i) for i in isa[0].layout.final_index_layout()] if isa[0].layout else None,
        quantum_seconds=usage, finished_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        mae_raw=float(np.abs(measured[:n] - exact[:n]).mean()), mae_fixed=float(np.abs(fixed - exact[:n]).mean()),
        mae_calibration_raw=float(np.abs(measured[n:] - exact[n:]).mean()),
        fixer_slope=float(fixer.coef_[0]), fixer_intercept=float(fixer.intercept_),
        fields=[dict(name=f['name'], sample_id=f['sample_id'], target=f['target'],
                     quantum_exact=str(v_exact[i]), quantum_hardware_raw=str(v_raw[i]), quantum_hardware_fixed=str(v_fixed[i]),
                     rbf=str(local[1, i]), forest=str(local[2, i]),
                     decision_exact=d_exact[i].tolist(), decision_raw=d_raw[i].tolist(), decision_fixed=d_fixed[i].tolist())
                for i, f in enumerate(fields)],
        exact=exact.tolist(), measured=measured.tolist(),
        scope='One small fixed-model inference example on a real device; not an accuracy estimate. No hardware training.')
    if save:
        save.parent.mkdir(exist_ok=True)
        save.write_text(json.dumps(out, indent=2))
    return out

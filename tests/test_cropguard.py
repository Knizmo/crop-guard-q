"""Self-checks for CropGuard-Q. Run from the repository root:  .venv/bin/python -m tests.test_cropguard"""
import itertools
import os

import numpy as np

from challenge import analyst, guard
from challenge.data import load_data
from challenge.hardware_kernel import quantum_votes
from challenge.kernel import QuantumKernelSVM, fidelity
from challenge.mitigation import circuit, ideal_zeros
from challenge.submission import LABELS, Committee, Submission


def test_closed_form_states_match_qiskit():
    q = QuantumKernelSVM(scale=0.3)
    X = np.random.default_rng(0).normal(size=(6, 8))
    assert np.allclose(q.states(X), q.qiskit_states(X), atol=1e-12)


def test_kernel_circuit_matches_exact_fidelity():
    q = QuantumKernelSVM(scale=0.075)
    a, b = np.random.default_rng(1).normal(size=(2, 8))
    assert np.isclose(ideal_zeros(circuit(q.scale * a, q.scale * b)), fidelity(q.states(a[None]), q.states(b[None]))[0, 0])


def test_vote_rule():
    """Any two agree -> that label; all three differ -> the RBF SVM (slot 1)."""
    for v in itertools.product(LABELS, repeat=3):
        V = np.array(v)[:, None]
        got = LABELS[Committee.vote_shares(V).argmax(axis=1)][0]
        want = v[0] if v[0] in (v[1], v[2]) else v[1]
        assert got == want, (v, got)


def test_submission_end_to_end_and_tamper_budget():
    tables, _, manifest, _ = load_data(); F = manifest['features']
    X, y = tables['train'][F].to_numpy(), tables['train'].target.to_numpy()
    Xv, yv = tables['validation'][F].to_numpy()[:20], tables['validation'].target.to_numpy()[:20]
    sub = Submission().fit(X, y, seed=0)
    p = sub.predict_proba(Xv)
    assert p.shape == (20, 3) and np.allclose(p.sum(axis=1), 1)
    assert (LABELS[p.argmax(axis=1)] == sub.predict(Xv)).all()
    assert (sub.predict(Xv[::-1])[::-1] == sub.predict(Xv)).all(), 'row order changed predictions'
    for bad in (Xv[:, :8], np.where(np.arange(13) == 0, np.nan, Xv)):
        try: sub.predict(bad); raise AssertionError('bad input accepted')
        except ValueError: pass
    adv = guard.tamper(guard.member_margin(sub, 1), Xv, guard.index(yv), 0.2, X, iters=30)
    assert (np.linalg.norm((adv - Xv) / X.std(axis=0), axis=1) <= 0.2 + 1e-9).all(), 'attack exceeded its budget'
    assert ((adv >= np.minimum(X.min(0), Xv) - 1e-12) & (adv <= np.maximum(X.max(0), Xv) + 1e-12)).all()
    # hardware path: support-vector-only kernel rows reproduce the exact decision
    q = sub.members[0]; Z = sub.transform(Xv[:3])
    d, v = quantum_votes(sub, fidelity(q.states(Z), q.states(q.train_X[q.svc.support_])), 3)
    assert np.allclose(d, q.decision_function(Z)) and (v == q.predict(Z)).all()


def test_analyst_is_safe_offline():
    info = dict(sample_id='r2_x', true='wheat', final='maize', flagged=True,
                votes={'quantum_svm': 'wheat', 'rbf_svm': 'maize', 'forest': 'maize'}, band_shift=[0.1] * 13, eps=0.1)
    os.environ.pop('CROPGUARD_LLM_URL', None)
    text, source = analyst.note(info)
    assert source == 'offline' and 'FLAGGED' in text
    os.environ['CROPGUARD_LLM_URL'] = 'http://example.com/v1/chat/completions'
    try:
        text, source = analyst.note(info)
        assert source == 'offline' and 'ValueError' in text, 'plain-http endpoint must be refused'
    finally:
        os.environ.pop('CROPGUARD_LLM_URL')


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('ok', name)

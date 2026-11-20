"""Attack lab and tamper alarm (additional analysis). Every attack works on copies of the data.

Tampering perturbs the raw measurements, not the PCA numbers, and the change passes through the
training-fitted preprocessing like a real edited record would. The budget `eps` is an L2 norm in
training-standardized band units (one unit = one training standard deviation of that band), and each
band stays inside the training range (or the field's own value), so a tampered field stays plausible.
The attack is a gradient-free random search: a weak attacker, so measured damage is a lower bound.
Default budgets, step counts and draws come from configs/experiment.json ("attack", "bootstrap", "seeds").
"""
import json

import numpy as np

from .config import EXPERIMENT
from .coverage import metric_vector
from .data import ROOT
from .metrics import score
from .submission import LABELS, MatchedControl, Submission, scores


def committees(X, y, seed=0):
    """The submitted committee and its matched control, fitted on identical rows (they share RBF SVM + forest)."""
    return Submission().fit(X, y, seed), MatchedControl().fit(X, y, seed)


def index(y):
    return np.searchsorted(LABELS, y)  # LABELS is sorted, like every sklearn classes_


def margin(S, yi):
    """True-class score minus best other score: > 0 correct, < 0 fooled."""
    ar = np.arange(len(yi))
    other = S.copy(); other[ar, yi] = -np.inf
    return S[ar, yi] - other.max(axis=1)


def member_margin(committee, slot):
    model = committee.members[slot]
    return lambda X, yi: margin(scores(model, committee.transform(X)), yi)


def committee_margin(committee, X0, yi):
    """Adaptive attacker: knows all three inspectors and pushes down the largest (scale-normalized) margin."""
    def margins(X, yi):
        Z = committee.transform(X)  # preprocess once per step, shared by the three inspectors
        return np.array([margin(scores(model, Z), yi) for model in committee.members])
    scale = np.median(np.abs(margins(X0, yi)), axis=1)[:, None]
    return lambda X, yi: (margins(X, yi) / scale).max(axis=0)


def tamper(margin_fn, X, yi, eps, X_train, iters=EXPERIMENT['attack']['iters'], seed=0):
    """Random search for a per-field change with standardized L2 norm <= eps that lowers margin_fn."""
    rng = np.random.default_rng(seed)
    sd = X_train.std(axis=0)
    lo, hi = np.minimum(X_train.min(axis=0), X), np.maximum(X_train.max(axis=0), X)
    apply = lambda D: np.clip(X + D * sd, lo, hi)  # clipping only moves towards X, so the budget still holds
    D, best = np.zeros_like(X), margin_fn(X, yi)
    for it in range(iters):
        cand = D + rng.normal(size=X.shape) * eps * (0.5 if it < iters // 2 else 0.15)
        cand *= np.minimum(1, eps / np.maximum(np.linalg.norm(cand, axis=1, keepdims=True), 1e-12))
        m = margin_fn(apply(cand), yi)
        better = m < best
        D[better], best = cand[better], np.where(better, m, best)
    return apply(D)


def slips(V, y):
    """Per-field outcomes for an alarm made of the inspectors in V (k, n)."""
    unanimous = (V == V[0]).all(axis=0)
    return dict(flagged=~unanimous, undetected=unanimous & (V[0] != y))


def alarm_views(sub, ctrl, X, y):
    """Votes of each alarm we compare. Slot 1 (RBF SVM) is the main inspector in all of them."""
    q, c = sub.votes(X), ctrl.votes(X)
    views = {'RBF SVM alone': q[[1]], '+ quantum SVM': q[[1, 0]], '+ poly SVM': c[[1, 0]], '+ forest': q[[1, 2]],
             'quantum committee': q, 'matched control': c}
    return {name: slips(V, y) for name, V in views.items()}, q, c


def tamper_study(X, y, Xv, yv, eps_list=EXPERIMENT['attack']['eps'], seeds=EXPERIMENT['seeds'], iters=EXPERIMENT['attack']['iters'], adaptive=True):
    """Naive attacker (aims at the shared RBF SVM) and adaptive attacker (knows the whole committee).
    Returns summary rows and per-field 'undetected' outcomes for the paired bootstrap."""
    yi, rows, fields = index(yv), [], {}
    for seed in seeds:
        sub, ctrl = committees(X, y, seed)
        clean, _, _ = alarm_views(sub, ctrl, Xv, yv)
        for name, s in clean.items():
            rows.append(dict(attack='none', eps=0, seed=seed, alarm=name, flagged=s['flagged'].mean(),
                             undetected=s['undetected'].mean()))
        for eps in eps_list:
            adv = tamper(member_margin(sub, 1), Xv, yi, eps, X, iters, seed)
            views, q, c = alarm_views(sub, ctrl, adv, yv)
            for name, s in views.items():
                rows.append(dict(attack='naive', eps=eps, seed=seed, alarm=name, flagged=s['flagged'].mean(),
                                 undetected=s['undetected'].mean()))
                fields[('naive', eps, seed, name)] = s['undetected']
            for name, model in [('quantum SVM', 0), ('RBF SVM', 1), ('forest', 2)]:
                rows.append(dict(attack='naive transfer', eps=eps, seed=seed, alarm=name, accuracy=(q[model] == yv).mean()))
            rows.append(dict(attack='naive transfer', eps=eps, seed=seed, alarm='poly SVM', accuracy=(c[0] == yv).mean()))
            for name, committee in [('quantum committee', sub), ('matched control', ctrl)] if adaptive else []:
                adv = tamper(committee_margin(committee, Xv, yi), Xv, yi, eps, X, iters, seed)
                r = committee.inspect(adv); s = slips(r['votes'], yv)
                rows.append(dict(attack='adaptive', eps=eps, seed=seed, alarm=name, flagged=s['flagged'].mean(),
                                 undetected=s['undetected'].mean(), accuracy=(r['prediction'] == yv).mean()))
                fields[('adaptive', eps, seed, name)] = s['undetected']
    return rows, fields


def poison_study(X, y, Xv, yv, rates=EXPERIMENT['attack']['poison_rates'], draws=EXPERIMENT['attack']['poison_draws']):
    """Flip a fraction of training labels to another crop at random, retrain everything, score on clean validation."""
    rows = []
    for rate in rates:
        for draw in range(draws):
            rng = np.random.default_rng(draw)
            yp = np.array(y, dtype=object)
            hit = rng.choice(len(y), int(rate * len(y)), replace=False)
            yp[hit] = [rng.choice([l for l in LABELS if l != old]) for old in yp[hit]]
            sub, ctrl = committees(X, yp.astype(str), seed=EXPERIMENT['seeds'][draw % len(EXPERIMENT['seeds'])])
            rows += f1_rows(sub, ctrl, Xv, yv, dict(rate=rate, draw=draw))
    return rows


def missing_study(sub, ctrl, X_train, Xv, yv, ks=EXPERIMENT['attack']['missing_bands'], draws=EXPERIMENT['attack']['missing_draws']):
    """k bands lost per field and imputed with the training mean, at prediction time only."""
    rows, mean = [], X_train.mean(axis=0)
    for k in ks:
        for draw in range(draws):
            rng = np.random.default_rng(draw)
            lost = np.argsort(rng.random(Xv.shape), axis=1)[:, :k]
            Xm = Xv.copy(); np.put_along_axis(Xm, lost, mean[lost], axis=1)
            rows += f1_rows(sub, ctrl, Xm, yv, dict(missing=k, draw=draw), alarm=True)
    return rows


def f1_rows(sub, ctrl, X, y, tags, alarm=False):
    q, c = sub.inspect(X), ctrl.inspect(X)
    out = {'quantum SVM': q['votes'][0], 'poly SVM': c['votes'][0], 'RBF SVM': q['votes'][1],
           'forest': q['votes'][2], 'quantum committee': q['prediction'], 'matched control': c['prediction']}
    rows = [dict(tags, model=name, macro_f1=score(y, p, list(LABELS))['macro_f1']) for name, p in out.items()]
    if alarm:
        for name, r in [('quantum committee', q), ('matched control', c)]:
            s = slips(r['votes'], y)
            rows.append(dict(tags, model=name + ' alarm', flagged=s['flagged'].mean(), undetected=s['undetected'].mean()))
    return rows


def block_weights(sample_ids, samples=EXPERIMENT['bootstrap']['samples'], seed=EXPERIMENT['bootstrap']['seed']):
    """Spatial-block bootstrap weights over validation fields (same scheme as challenge/coverage.py)."""
    groups = json.loads((ROOT / 'data/validation_groups.json').read_text())
    unique, idx = np.unique([groups[s] for s in sample_ids], return_inverse=True)
    rng = np.random.default_rng(seed)
    return np.array([np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))[idx]
                     for _ in range(samples)])


def paired_bootstrap(a, b, y, weights, stat='macro_f1'):
    """Difference b - a on the same resampled spatial blocks. a, b: (seeds, n) labels, or 0/1 per-field outcomes."""
    if stat == 'macro_f1':
        yi = index(y)
        f = lambda P, w: metric_vector(yi, index(P), w)[0]
    else:
        f = lambda P, w: float((np.asarray(P, float) * w).sum(axis=-1).mean() / w.sum())
    boot = np.array([f(b, w) - f(a, w) for w in weights])
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return dict(statistic=stat, delta=f(b, np.ones(len(y))) - f(a, np.ones(len(y))), ci95=[float(lo), float(hi)],
                share_positive=float((boot > 0).mean()), resamples=len(weights),
                method=f"paired spatial-block bootstrap (data/validation_groups.json), seed {EXPERIMENT['bootstrap']['seed']}")

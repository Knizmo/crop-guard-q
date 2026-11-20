"""Validation grid search with an equal budget: 30 configurations for every inspector, then frozen.

Run from the repository root:  .venv/bin/python -m challenge.tune
Writes configs/cropguard.json (frozen settings) and outputs/tuning_log.json (every configuration tried).
Selection uses validation macro-F1 (forest: seed 0); ties go to the earlier, simpler grid entry.
"""
import json
from itertools import product
from time import perf_counter

from .data import ROOT, load_data
from .metrics import score
from .submission import LABELS, member, preprocessing

GRIDS = {
    'quantum': dict(scale=[0.05, 0.075, 0.1, 0.15, 0.2, 0.3], C=[1, 3, 10, 30, 100]),
    'poly': dict(degree=[2, 3, 4], coef0=[1, 2], C=[0.3, 1, 3, 10, 30]),
    'rbf': dict(gamma=[0.03, 0.06, 0.125, 0.25, 0.5, 1.0], C=[1, 3, 10, 30, 100]),
    'forest': dict(max_features=[1, 2, 3, 4, 6, 8], min_samples_leaf=[1, 2, 3, 5, 8]),
}


def grid(name):
    keys, values = zip(*GRIDS[name].items())
    return [dict(zip(keys, v)) for v in product(*values)]


def tune(write=True):
    """write=False reruns the search without touching the frozen configs/cropguard.json."""
    tables, _, manifest, _ = load_data()
    F, train, val = manifest['features'], tables['train'], tables['validation']
    prep = preprocessing().fit(train[F].to_numpy())
    Z, Zv = prep.transform(train[F].to_numpy()), prep.transform(val[F].to_numpy())
    config, log = {}, {}
    for name in GRIDS:
        start, rows = perf_counter(), []
        for params in grid(name):
            model = member(name, params, seed=0).fit(Z, train.target)
            rows.append({**params, 'validation_macro_f1': score(val.target, model.predict(Zv), LABELS)['macro_f1']})
        best = max(rows, key=lambda r: r['validation_macro_f1'])
        config[name] = {k: v for k, v in best.items() if k != 'validation_macro_f1'}
        log[name] = dict(configurations=len(rows), seconds=perf_counter() - start,
                         best_validation_macro_f1=best['validation_macro_f1'], rows=rows)
        print(f'{name:8s} {len(rows)} configs  {log[name]["seconds"]:5.1f}s  best {config[name]}'
              f'  F1 {best["validation_macro_f1"]:.4f}', flush=True)
    config['selection'] = ('Validation macro-F1 over an equal 30-configuration grid per inspector '
                           '(challenge/tune.py); frozen before any test prediction.')
    if write: (ROOT / 'configs/cropguard.json').write_text(json.dumps(config, indent=2) + '\n')
    (ROOT / 'outputs').mkdir(exist_ok=True)
    (ROOT / 'outputs/tuning_log.json').write_text(json.dumps(log, indent=2))
    return config, log


if __name__ == '__main__':
    tune()

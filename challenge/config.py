"""Every setting in one place, read from files instead of typed into the code.

MANIFEST    data/manifest.json: crop labels, feature names, dates, split sizes (the released data's own facts)
EXPERIMENT  configs/experiment.json: study parameters (qubits, grids, attack budgets, noise and hardware runs)
frozen()    configs/cropguard.json: the settings tuning selected (written by challenge/tune.py)
"""
import json

import numpy as np

from .data import ROOT

MANIFEST = json.loads((ROOT / 'data/manifest.json').read_text())
EXPERIMENT = json.loads((ROOT / 'configs/experiment.json').read_text())
LABELS = np.array(MANIFEST['labels'])
FEATURES = list(MANIFEST['features'])
QUBITS = int(EXPERIMENT['qubits'])
OUT = ROOT / 'outputs'

if not 1 <= QUBITS <= 8: raise ValueError('The challenge allows at most 8 qubits per circuit')
if QUBITS > len(FEATURES): raise ValueError('Cannot keep more PCA components than there are features')
# sklearn orders classes alphabetically; votes, scores and margins rely on the same order
if list(LABELS) != sorted(LABELS): raise ValueError('Manifest labels must be in alphabetical order')


def frozen():
    return json.loads((ROOT / 'configs/cropguard.json').read_text())

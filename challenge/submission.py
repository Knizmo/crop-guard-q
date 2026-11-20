"""CropGuard-Q submission: three inspectors vote on each field and a disagreement raises a tamper flag.

Input: the raw features in data/manifest.json order (B1 ... B12, NDVI); IDs are never inputs.
`fit` learns StandardScaler -> PCA(qubits) -> StandardScaler on the training rows only (with 8 qubits this is
the recipe of data/views/pca8.npz), so all inspectors see the same numbers, one per qubit:
  slot 0  partner under study: the quantum-kernel SVM (Submission) or a polynomial SVM (MatchedControl)
  slot 1  classical RBF SVM
  slot 2  random forest (the only stochastic member: seed 0, 1, 2)
Label = majority vote; if all three differ, the RBF SVM decides. Frozen settings: configs/cropguard.json;
qubits and forest size: configs/experiment.json (see challenge/config.py).
"""
from time import perf_counter

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import EXPERIMENT, FEATURES, LABELS, QUBITS, frozen as load_config
from .kernel import QuantumKernelSVM
from .metrics import probabilities

VOTE_WEIGHTS = np.array([1, 1.001, 1])  # equal votes; the tiny extra lets the RBF SVM settle a three-way split


def preprocessing():
    return make_pipeline(StandardScaler(), PCA(QUBITS, svd_solver='full'), StandardScaler())


def member(name, params, seed=0):
    if name == 'quantum': return QuantumKernelSVM(**params)
    if name == 'rbf': return SVC(kernel='rbf', **params)
    if name == 'poly': return SVC(kernel='poly', gamma='scale', **params)
    if name == 'forest': return RandomForestClassifier(n_estimators=EXPERIMENT['forest_trees'], random_state=seed, **params)
    raise ValueError(f'Unknown inspector {name}')


def scores(model, Z):
    """Per-class scores (n, classes): SVM one-vs-rest decision values, forest probabilities."""
    return model.predict_proba(Z) if isinstance(model, RandomForestClassifier) else model.decision_function(Z)


def raw_features(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(FEATURES) or not np.isfinite(X).all():
        raise ValueError(f'Expected a finite (n, {len(FEATURES)}) array of {", ".join(FEATURES)}')
    return X


class Committee:
    def __init__(self, partner='quantum', config=None):
        self.config = config or load_config()
        self.names = (partner, 'rbf', 'forest')
        self.labels = LABELS

    def fit(self, X_train, y_train, seed=0):
        X, y = raw_features(X_train), np.asarray(y_train)
        if set(y) != set(LABELS): raise ValueError('Training must contain every declared class')
        start = perf_counter()
        self.prep = preprocessing().fit(X)
        Z = self.prep.transform(X)
        self.fit_seconds = {'preprocessing': perf_counter() - start}
        self.members = [member(name, self.config[name], seed) for name in self.names]
        for name, model in zip(self.names, self.members):
            t = perf_counter(); model.fit(Z, y); self.fit_seconds[name] = perf_counter() - t
        self.fit_seconds['total'] = perf_counter() - start
        return self

    def transform(self, X):
        return self.prep.transform(raw_features(X))

    def votes(self, X):
        """(3, n) labels in slot order [partner, rbf, forest]."""
        Z = self.transform(X)
        return np.stack([model.predict(Z) for model in self.members])

    @staticmethod
    def vote_shares(V):
        return (VOTE_WEIGHTS[:, None, None] * (V[:, :, None] == LABELS)).sum(axis=0) / VOTE_WEIGHTS.sum()

    def predict_proba(self, X):
        """Weighted vote shares; argmax equals predict(). Not calibrated crop probabilities."""
        return probabilities(self.vote_shares(self.votes(X)), len(X), len(LABELS))

    def predict(self, X):
        return LABELS[self.predict_proba(X).argmax(axis=1)]

    def inspect(self, X):
        """Everything one pass gives: votes, final label and the tamper flag (True = not unanimous)."""
        V = self.votes(X)
        return dict(votes=V, prediction=LABELS[self.vote_shares(V).argmax(axis=1)], flag=~(V == V[0]).all(axis=0))

    def flag(self, X):
        return self.inspect(X)['flag']

    def resources(self):
        scaler1, pca, scaler2 = self.prep
        out = dict(inputs=f'{len(FEATURES)} raw features -> {pca.n_components_} train-fitted PCA numbers',
                   preprocessing_fitted_values=int(scaler1.mean_.size + scaler1.scale_.size + pca.components_.size
                                                   + pca.mean_.size + scaler2.mean_.size + scaler2.scale_.size),
                   fit_seconds=self.fit_seconds, members={})
        for name, model in zip(self.names, self.members):
            if name == 'quantum':
                n, sv = len(model.train_states), int(model.svc.n_support_.sum())
                r = model.resources()
                r.update(training_states_simulated=n, training_kernel_circuits=n * (n - 1) // 2,
                         circuits_per_predicted_field=sv, kernel_storage_bytes=n * n * np.dtype(float).itemsize,
                         dual_coefficients=int(model.svc.dual_coef_.size), simulation='exact statevector')
            elif name == 'forest':
                nodes = [t.tree_.node_count for t in model.estimators_]
                r = dict(trees=len(nodes), total_nodes=int(sum(nodes)),
                         mean_depth=float(np.mean([t.tree_.max_depth for t in model.estimators_])))
            else:
                r = dict(support_vectors=int(model.n_support_.sum()), stored_values=int(model.support_vectors_.size
                                                                                       + model.dual_coef_.size))
            out['members'][name] = {**r, 'params': self.config[name]}
        return out


class Submission(Committee):
    """Final model: the quantum-kernel SVM takes part in every prediction."""
    def __init__(self, config=None):
        super().__init__('quantum', config)


class MatchedControl(Committee):
    """Required control: the same pipeline with the quantum member replaced by a polynomial-kernel SVM."""
    def __init__(self, config=None):
        super().__init__('poly', config)

"""Quantum kernel (ZZ feature map, exact statevectors) + SVM; one qubit per prepared feature."""
from functools import cache

import numpy as np
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector
from sklearn.svm import SVC


@cache
def basis(n):
    """Basis bitstrings (Qiskit little-endian order) and neighbour parities for n qubits."""
    bits = (np.arange(2 ** n)[:, None] >> np.arange(n)) & 1
    return bits, bits[:, :-1] ^ bits[:, 1:]


def fidelity(A, B):
    """Kernel matrix |<a|b>|^2 between two stacks of statevectors."""
    return np.abs(A.conj() @ B.T) ** 2


class QuantumKernelSVM:
    def __init__(self, scale=0.1, C=10, reps=1):
        self.scale = scale  # data-rotation strength: too large collapses the kernel
        self.reps = reps
        self.svc = SVC(kernel='precomputed', C=C)

    def feature_map(self, n):
        return zz_feature_map(n, reps=self.reps, entanglement='linear')

    def states(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not 1 <= X.shape[1] <= 8 or not np.isfinite(X).all():
            raise ValueError('Expected a finite (n, qubits) prepared view with at most 8 columns')
        if hasattr(self, 'n') and X.shape[1] != self.n:
            raise ValueError(f'Model was fitted on {self.n} features, got {X.shape[1]}')
        if self.reps != 1:
            return self.qiskit_states(X)
        # reps=1 ZZ map in closed form: H on every qubit, then diagonal phases exp(i*phi(b)).
        # Identical to Qiskit's Statevector (checked in tests/test_cropguard.py), ~100x faster.
        bits, pairs = basis(X.shape[1])
        x = self.scale * X
        phase = 2 * x @ bits.T + 2 * ((np.pi - x[:, :-1]) * (np.pi - x[:, 1:])) @ pairs.T
        return np.exp(1j * phase) / np.sqrt(len(bits))

    def qiskit_states(self, X):
        fm = self.feature_map(np.asarray(X).shape[1])
        return np.array([Statevector(fm.assign_parameters(self.scale * x)).data for x in np.asarray(X)])

    def fit(self, X, y):
        self.train_X = np.asarray(X, dtype=float)  # kept to build hardware circuits against the support vectors
        self.n = self.train_X.shape[1]
        self.train_states = self.states(self.train_X)
        self.svc.fit(fidelity(self.train_states, self.train_states), y)
        self.classes_ = self.svc.classes_
        self.sv_states = self.train_states[self.svc.support_]
        return self

    def kernel(self, X):
        """Kernel rows against the training set. Only support-vector columns enter the SVM decision,
        so only those overlaps are computed; the other columns stay 0."""
        K = np.zeros((len(X), len(self.train_states)))
        K[:, self.svc.support_] = fidelity(self.states(X), self.sv_states)
        return K

    def predict(self, X):
        return self.svc.predict(self.kernel(X))

    def decision_function(self, X):
        return self.svc.decision_function(self.kernel(X))

    def pair_circuit(self, x, y):
        """Compute-uncompute circuit a QPU runs: P(all zeros) = |<phi(y)|phi(x)>|^2."""
        fm = self.feature_map(len(x))
        qc = fm.assign_parameters(self.scale * np.asarray(x))
        qc = qc.compose(fm.assign_parameters(self.scale * np.asarray(y)).inverse())
        qc.measure_all()
        return qc

    def resources(self):
        decomposed = self.feature_map(self.n).decompose()
        cx = decomposed.count_ops().get('cx', 0)
        return dict(qubits=self.n, feature_map_depth=decomposed.depth(), feature_map_cx=cx, kernel_circuit_cx=2 * cx,
                    trainable_quantum=0, support_vectors=int(self.svc.n_support_.sum()))

"""8-qubit quantum kernel (ZZ feature map, exact statevectors) + SVM on the 8-feature PCA view."""
import numpy as np
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector
from sklearn.svm import SVC

N = 8
BITS = (np.arange(2 ** N)[:, None] >> np.arange(N)) & 1  # basis states, Qiskit little-endian order
PAIRS = BITS[:, :-1] ^ BITS[:, 1:]  # parity of neighbouring qubits (linear entanglement)


def fidelity(A, B):
    """Kernel matrix |<a|b>|^2 between two stacks of statevectors."""
    return np.abs(A.conj() @ B.T) ** 2


class QuantumKernelSVM:
    def __init__(self, scale=0.1, C=10, reps=1):
        self.scale = scale  # data-rotation strength: the default 1.0 collapses macro-F1 to ~0.3
        self.reps = reps
        self.feature_map = zz_feature_map(N, reps=reps, entanglement='linear')
        self.svc = SVC(kernel='precomputed', C=C)

    def states(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != N or not np.isfinite(X).all():
            raise ValueError('Expected a finite (n,8) prepared view')
        if self.reps != 1:
            return self.qiskit_states(X)
        # reps=1 ZZ map in closed form: H on every qubit, then diagonal phases exp(i*phi(b)).
        # Identical to Qiskit's Statevector (checked in tests/test_cropguard.py), ~100x faster.
        x = self.scale * X
        phase = 2 * x @ BITS.T + 2 * ((np.pi - x[:, :-1]) * (np.pi - x[:, 1:])) @ PAIRS.T
        return np.exp(1j * phase) / 2 ** (N / 2)

    def qiskit_states(self, X):
        return np.array([Statevector(self.feature_map.assign_parameters(self.scale * x)).data for x in np.asarray(X)])

    def fit(self, X, y):
        self.train_X = np.asarray(X, dtype=float)  # kept to build hardware circuits against the support vectors
        self.train_states = self.states(X)
        self.svc.fit(fidelity(self.train_states, self.train_states), y)
        self.classes_ = self.svc.classes_
        return self

    def kernel(self, X):
        return fidelity(self.states(X), self.train_states)

    def predict(self, X):
        return self.svc.predict(self.kernel(X))

    def decision_function(self, X):
        return self.svc.decision_function(self.kernel(X))

    def pair_circuit(self, x, y):
        """Compute-uncompute circuit a QPU runs: P(all zeros) = |<phi(y)|phi(x)>|^2."""
        qc = self.feature_map.assign_parameters(self.scale * np.asarray(x))
        qc = qc.compose(self.feature_map.assign_parameters(self.scale * np.asarray(y)).inverse())
        qc.measure_all()
        return qc

    def resources(self):
        decomposed = self.feature_map.decompose()
        return dict(qubits=N, feature_map_depth=decomposed.depth(), feature_map_cx=decomposed.count_ops().get('cx', 0),
                    kernel_circuit_cx=2 * decomposed.count_ops().get('cx', 0), trainable_quantum=0,
                    support_vectors=int(self.svc.n_support_.sum()))

"""8-qubit quantum kernel (ZZ feature map, exact statevectors) + SVM on the 8-feature PCA view."""
import numpy as np
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector
from sklearn.svm import SVC


def fidelity(A, B):
    """Kernel matrix |<a|b>|^2 between two stacks of statevectors."""
    return np.abs(A.conj() @ B.T) ** 2


class QuantumKernelSVM:
    def __init__(self, scale=0.1, C=10, reps=1):
        self.scale = scale  # data-rotation strength: the default 1.0 collapses macro-F1 to ~0.3
        self.feature_map = zz_feature_map(8, reps=reps, entanglement='linear')
        self.svc = SVC(kernel='precomputed', C=C)

    def states(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != 8 or not np.isfinite(X).all():
            raise ValueError('Expected a finite (n,8) prepared view')
        return np.array([Statevector(self.feature_map.assign_parameters(self.scale * x)).data for x in X])

    def fit(self, X, y):
        self.train_states = self.states(X)
        self.svc.fit(fidelity(self.train_states, self.train_states), y)
        return self

    def predict(self, X):
        return self.svc.predict(fidelity(self.states(X), self.train_states))

    def pair_circuit(self, x, y):
        """Compute-uncompute circuit a QPU runs: P(all zeros) = |<phi(y)|phi(x)>|^2."""
        qc = self.feature_map.assign_parameters(self.scale * np.asarray(x))
        qc = qc.compose(self.feature_map.assign_parameters(self.scale * np.asarray(y)).inverse())
        qc.measure_all()
        return qc

    def resources(self):
        ops = self.feature_map.count_ops()
        return dict(qubits=8, depth=self.feature_map.depth(), two_qubit_gates=ops.get('cx', 0),
                    trainable_quantum=0, support_vectors=int(self.svc.n_support_.sum()))

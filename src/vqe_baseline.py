import numpy as np
import mlflow

from qiskit.circuit.library import TwoLocal
from qiskit.primitives import StatevectorEstimator
#from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp

from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA

# Seed 고정
np.random.seed(42)

# 2-qubit Hamiltonian (Z ⊗ Z)
H = SparsePauliOp.from_list([("ZZ", 1.0)])

# Ansatz
ansatz = TwoLocal(num_qubits=2, rotation_blocks="ry", entanglement_blocks="cz")

# Optimizer
optimizer = COBYLA(maxiter=100)

# Estimator (new primitive)
estimator = StatevectorEstimator()

# MLflow experiment
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("vqe_baseline")



with mlflow.start_run():
    mlflow.log_param("optimizer", "COBYLA")
    mlflow.log_param("maxiter", 100)

    vqe = VQE(estimator, ansatz, optimizer)
    result = vqe.compute_minimum_eigenvalue(H)

    energy = result.eigenvalue.real

    mlflow.log_metric("energy", energy)

    print("Minimum energy:", energy)

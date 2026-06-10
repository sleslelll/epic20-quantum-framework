import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA


# =========================
# 1. OFDM 채널 생성
# =========================

def generate_ofdm_channel(n=2):
    # n x n 랜덤 채널 행렬 생성
    H = np.random.randn(n, n)
    return H


# =========================
# 2. 송신 신호 생성
# =========================

def generate_signal(n=2):
    # BPSK 스타일 (-1, 1) 신호 생성
    x_true = np.random.choice([-1, 1], size=n)
    return x_true


# =========================
# 3. 채널 통과 (전송 모델)
# =========================

def transmit(H, x):
    # y = Hx
    return H @ x


# =========================
# 4. 고전적 MMSE 추정
# =========================

def mmse_estimate(H, y, noise=1e-3):
    # (H^T H + λI)^(-1) H^T y
    HtH = H.T @ H + noise * np.eye(H.shape[1])
    return np.linalg.inv(HtH) @ H.T @ y


# =========================
# 5. 비용 함수 정의
# =========================

def cost_function(x, H, y):
    # ||Hx - y||^2
    return np.linalg.norm(H @ x - y) ** 2


# =========================
# 6. Hamiltonian 구성 (단순화 버전)
# =========================

def build_hamiltonian(H, y):
    """
    최소화: ||Hx - y||²
    
    전개: x^T H^T H x - 2 y^T H x + y^T y
    """
    n = len(y)
    G = H.T @ H
    Hy = H.T @ y
    
    pauli_list = []
    
    # 대각 항
    for i in range(n):
        pauli_str = ["I"] * n
        pauli_str[i] = "Z"
        pauli_list.append(("".join(pauli_str), float(G[i, i])))
    
    # 비대각 항
    for i in range(n):
        for j in range(i+1, n):
            pauli_str = ["I"] * n
            pauli_str[i] = "Z"
            pauli_str[j] = "Z"
            pauli_list.append(("".join(pauli_str), float(2*G[i, j])))
    
    # Linear 항
    for i in range(n):
        pauli_str = ["I"] * n
        pauli_str[i] = "Z"
        pauli_list.append(("".join(pauli_str), float(-2.0 * Hy[i])))
    
    return SparsePauliOp.from_list(pauli_list)

# =========================
# 7. Ansatz 정의
# =========================

def build_ansatz(num_qubits):
    # Variational circuit (parameterized)
    return TwoLocal(
        num_qubits=num_qubits,
        rotation_blocks="ry",
        entanglement_blocks="cz",
        reps=2
    )


# =========================
# 8. VQE 실행
# =========================

def run_vqe(hamiltonian, num_qubits):
    # Ansatz 생성
    ansatz = build_ansatz(num_qubits)

    # 최적화 알고리즘
    optimizer = COBYLA(maxiter=200)

    # 상태 벡터 기반 estimator
    estimator = StatevectorEstimator()

    # VQE 객체 생성
    vqe = VQE(estimator, ansatz, optimizer)

    # 최소 고유값 계산
    result = vqe.compute_minimum_eigenvalue(hamiltonian)

    return result


# =========================
# 9. 디코딩
# =========================

def decode_vqe_to_signal(result, ansatz, num_qubits):
    """VQE 최적 상태에서 신호 추출"""
    from qiskit.quantum_info import Statevector
    
    optimal_circuit = ansatz.assign_parameters(result.optimal_point)
    state = Statevector(optimal_circuit)
    
    x_vqe = []
    for i in range(num_qubits):
        pauli_str = ["I"] * num_qubits
        pauli_str[i] = "Z"
        obs = SparsePauliOp.from_list([("".join(pauli_str), 1.0)])
        
        z_exp = state.expectation_value(obs).real
        x_i = 1 if z_exp > 0 else -1
        x_vqe.append(x_i)
    
    return np.array(x_vqe)


# =========================
# 10. 전체 파이프라인 실행
# =========================

def run_vqe_ofdm_pipeline():

    print("=== OFDM VQE 파이프라인 시작 ===")

    # 1. 채널 및 신호 생성
    np.random.seed(42)
    H = generate_ofdm_channel(n=2)
    x_true = generate_signal(n=2)


    # 2. 채널 통과
    y = transmit(H, x_true)

    print("채널 H:\n", H)
    print("원본 신호:", x_true)
    print("수신 신호 y:", y)


    # 3. 고전적 해 (MMSE)
    x_mmse = mmse_estimate(H, y)
    print("\n=== 고전적 MMSE 결과 ===")
    print("추정 신호:", x_mmse)
    mmse_error = cost_function(x_mmse, H, y)
    print("오차:", mmse_error)


    # 4. 양자 문제 구성 및 VQE 실행
    print("\n=== VQE 실행 ===")
    hamiltonian = build_hamiltonian(H, y)
    num_qubits = 2

    ansatz = build_ansatz(num_qubits)
    optimizer = COBYLA(maxiter=200)
    estimator = StatevectorEstimator()
    vqe = VQE(estimator, ansatz, optimizer)
    result = vqe.compute_minimum_eigenvalue(hamiltonian)


    # 5. 디코딩 *this is necessary to see the vqe error
    x_vqe = decode_vqe_to_signal(result, ansatz, num_qubits)
    vqe_error = cost_function(x_vqe, H, y)
    
    print("\n=== 양자 VQE 결과 ===")
    print("추정 신호:", x_vqe)
    print("오차:", vqe_error)
    

    # 6. 비교
    print("\n=== 결과 비교 ===")
    print(f"원본:   {x_true}")
    print(f"MMSE:   {x_mmse.round(2)} (오차: {mmse_error:.6f})")
    print(f"VQE:    {x_vqe} (오차: {vqe_error:.6f})")
    
    print("\n=== 파이프라인 종료 ===")


if __name__ == "__main__":

    run_vqe_ofdm_pipeline()
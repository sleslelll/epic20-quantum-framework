import numpy as np
import mlflow
import json
from pathlib import Path

from qiskit.circuit.library import TwoLocal, RealAmplitudes
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


import os
from pathlib import Path

# ══════════════════════════════════════════
# 경로 설정: 항상 quantum-research-0/ 기준
# ══════════════════════════════════════════

# 이 파일의 위치: quantum-research-0/src/vqe_signal_recovery.py
SCRIPT_DIR = Path(__file__).resolve().parent  # src/
PROJECT_ROOT = SCRIPT_DIR.parent               # quantum-research-0/

# 작업 디렉토리를 프로젝트 루트로 변경
os.chdir(PROJECT_ROOT)


# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────

SIGNAL_TYPES = ["audio", "image", "mri"]  # 3가지 예시
REPS = 2
OPTIMIZER_TYPE = "COBYLA"


# ─────────────────────────────────────────
# 신호 복원 Hamiltonian
# ─────────────────────────────────────────

# 2. Hamiltonian

def get_signal_recovery_hamiltonian(
    signal_observed,  # 관측된 신호
    mask,             # 어느 부분이 관측되었는지
    lambda_sparse=0.1 # 희소성 가중치
):
    """
    신호 복원을 위한 Hamiltonian
    
    H = Σ (x_i - y_i)² + λ Σ |x_i - x_{i+1}|
        (관측 위치)      (smoothness)
    """
    n_pixels = len(signal_observed.flatten())
    pauli_list = []
    
    # Data fidelity term
    for i, (val, observed) in enumerate(zip(
        signal_observed.flatten(), 
        mask.flatten()
    )):
        if observed:
            # (Z_i - val)² 근사
            pauli_list.append(("Z", -2.0 * val))   # 단일 큐비트 "Z"
            # pauli_list.append((f"Z{i}", -2.0 * val))
            pauli_list.append(("I" * n_pixels, val**2))
    

    # Smoothness term 
    for i in range(n_pixels - 1):
        pauli_str = ["I"] * n_pixels
        pauli_str[i] = "Z"
        pauli_str[i+1] = "Z"
        pauli_list.append(("".join(pauli_str), -lambda_sparse))
    
    return SparsePauliOp.from_list(pauli_list)


# ─────────────────────────────────────────
# VQE 실행
# ─────────────────────────────────────────

# 3. VQE 실행
def run_vqe_signal_recovery(
    signal_type,      # "audio", "image", "mri"
    sampling_rate=0.5 # 몇 % 샘플링
):
    """신호 복원 VQE 실행"""
    
    # 1. 신호 생성
    if signal_type == "audio":
        signal_clean = generate_audio_signal()
    elif signal_type == "image":
        signal_clean = generate_mnist_digit()
    elif signal_type == "mri":
        signal_clean = generate_mri_phantom()
    
    # 2. Undersampling
    mask = np.random.rand(*signal_clean.shape) < sampling_rate
    signal_observed = signal_clean * mask
    
    # 3. Hamiltonian
    H = get_signal_recovery_hamiltonian(signal_observed, mask)
    
    # 4. VQE
    n_qubits = signal_clean.size
    ansatz = RealAmplitudes(num_qubits=n_qubits, reps=REPS)
    optimizer = COBYLA(maxiter=200)
    estimator = StatevectorEstimator()
    
    vqe = VQE(estimator, ansatz, optimizer)
    result = vqe.compute_minimum_eigenvalue(H)
    
    # 5. 결과 해석
    signal_recovered = decode_from_vqe(result.optimal_point, signal_clean.shape)
    
    # 6. 평가
    psnr = calculate_psnr(signal_clean, signal_recovered)
    
    return {
        "signal_type": signal_type,
        "psnr": psnr,
        "num_iterations": result.optimizer_result.nfev,
        "signal_clean": signal_clean,
        "signal_observed": signal_observed,
        "signal_recovered": signal_recovered,
    }


# ─────────────────────────────────────────
# 신호 생성 함수들
# ─────────────────────────────────────────
# 4. 신호 생성


# Audio: 8 픽셀 (8 큐비트)
def generate_audio_signal(n=8):  # ← 8로 변경

    t = np.linspace(0, 1, n)
    signal = (np.sin(2*np.pi*5*t) + 
              0.5*np.sin(2*np.pi*13*t) + 
              0.3*np.sin(2*np.pi*20*t))
    return signal


# MNIST: 4×4 = 16 픽셀 (16 큐비트)
def generate_mnist_digit(digit=0):
    from sklearn.datasets import load_digits
    digits = load_digits()
    return digits.images[digit][:4, :4]  # ← 8x8 → 4x4


# MRI: 6×6 = 36 픽셀 (36 큐비트)
def generate_mri_phantom():
    from skimage.data import shepp_logan_phantom
    phantom = shepp_logan_phantom()
    return phantom[::64, ::64]  # ← ::32 → ::64 (더 작게)


# ─────────────────────────────────────────
# VQE 결과 디코딩 & 평가
# ─────────────────────────────────────────

def decode_from_vqe(optimal_point, original_shape):
    """
    VQE 최적 파라미터 → 복원 신호
    """
    n_pixels = np.prod(original_shape)
    
    # 파라미터를 신호 길이에 맞춤
    if len(optimal_point) < n_pixels:
        # 파라미터가 부족하면 반복
        signal_flat = np.tile(
            optimal_point, 
            int(np.ceil(n_pixels / len(optimal_point)))
        )[:n_pixels]
    else:
        # 파라미터가 많으면 앞부분만
        signal_flat = optimal_point[:n_pixels]
    
    # 정규화 (-1~1 범위 → 0~1 범위)
    signal_flat = (signal_flat - signal_flat.min()) / (
        signal_flat.max() - signal_flat.min() + 1e-10
    )
    
    # 원래 shape으로 복원
    return signal_flat.reshape(original_shape)


def calculate_psnr(signal_true, signal_recovered):
    """
    Peak Signal-to-Noise Ratio 계산
    
    PSNR = 20 * log10(MAX / sqrt(MSE))
    
    높을수록 복원 품질 좋음 (보통 20~40 dB)
    """
    mse = np.mean((signal_true - signal_recovered) ** 2)
    
    if mse == 0:
        return float('inf')  # 완벽한 복원
    
    max_val = max(signal_true.max(), signal_recovered.max())
    psnr = 20 * np.log10(max_val / np.sqrt(mse))
    
    return psnr



# ─────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────
# 5. 시각화

def plot_signal_recovery(results, save_path):
    """신호 복원 결과 시각화"""
    signal_type = results["signal_type"]
    
    if signal_type == "audio":
        # 1D 파형 3개
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        # ... (원본, 관측, 복원)
    
    elif signal_type in ["image", "mri"]:
        # 2D 이미지 3개
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        # ... (원본, 관측, 복원)
    
    plt.savefig(save_path, dpi=150)
    plt.close()



# ─────────────────────────────────────────
# 메인
# ─────────────────────────────────────────

if __name__ == "__main__":

    # MLflow 설정
    mlruns_path = PROJECT_ROOT / "mlruns"
    results_path = PROJECT_ROOT / "results" / "signal_recovery"
    
    print(f"MLflow 경로 설정: {mlruns_path}")
    mlflow.set_tracking_uri(f"file://{mlruns_path}")
    
    print(f"실험 이름: vqe_signal_recovery")
    mlflow.set_experiment("vqe_signal_recovery")
    
    print(f"결과 폴더: {results_path}")
    results_path.mkdir(parents=True, exist_ok=True)
    
    print()
    print("=" * 60)
    print("설정 완료, 실행 시작...")
    print("=" * 60)
    print()

    mlflow.set_tracking_uri(f"file://{PROJECT_ROOT}/mlruns")
    mlflow.set_experiment("vqe_signal_recovery")

    results_dir = Path("results/signal_recovery")
    results_dir.mkdir(parents=True, exist_ok=True)
    

    # 3가지 예시 실행
    for signal_type in SIGNAL_TYPES:
        
        with mlflow.start_run(run_name=f"{signal_type}_recovery"):
            
            results = run_vqe_signal_recovery(signal_type)
            
            mlflow.log_param("signal_type", signal_type)
            mlflow.log_param("sampling_rate", 0.5)
            mlflow.log_metric("psnr", results["psnr"])
            
            # 그래프 저장
            graph_path = results_dir / f"{signal_type}_recovery.png"
            plot_signal_recovery(results, graph_path)
            mlflow.log_artifact(str(graph_path))
            
            print(f"{signal_type}: PSNR = {results['psnr']:.2f} dB")

import numpy as np
import mlflow
import json

from pathlib import Path

from qiskit.circuit.library import TwoLocal
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA, SPSA

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


REPS=2

# OPTIMIZER = SPSA(maxiter=200)
# OPTIMIZER = COBYLA

OPTIMIZER_MAP = {
    "COBYLA": COBYLA,
    "SPSA": SPSA,
}

OPTIMIZER_TYPE = "COBYLA"


# optimizer_type = "SPSA"

# ──────────────────────────────────────────
# H₂ 해밀토니안 계수 (문헌값)
# O'Malley et al., PRX 6, 031007 (2016)
# STO-3G basis, Jordan-Wigner mapping
# H = g0*II + g1*Z0 + g2*Z1 + g3*Z0Z1 + g4*(X0X1 + Y0Y1)
# ──────────────────────────────────────────

# H=g0​I+g1​Z0​+g2​Z1​+g3​Z0​Z1​+g4​(X0​X1​+Y0​Y1​)
# “Quantum Chemistry Benchmark 기반 VQE Validation Framework”

H2_COEFFS = {
    # dist:  g0       g1       g2       g3       g4
    0.20: (-0.4347,  0.3435,  -0.4347,  0.5716,  0.0910),
    0.30: (-0.6656,  0.2835,  -0.4931,  0.5253,  0.1201),
    0.40: (-0.8246,  0.2395,  -0.5231,  0.4945,  0.1418),
    0.50: (-0.9443,  0.2086,  -0.5390,  0.4716,  0.1566),
    0.60: (-1.0336,  0.1855,  -0.5450,  0.4534,  0.1659),
    0.70: (-1.0987,  0.1685,  -0.5449,  0.4380,  0.1707),
    0.735:(-1.1168,  0.1633,  -0.5440,  0.4326,  0.1718),
    0.80: (-1.1498,  0.1565,  -0.5416,  0.4251,  0.1720),
    0.90: (-1.1855,  0.1469,  -0.5365,  0.4099,  0.1702),
    1.00: (-1.2088,  0.1399,  -0.5308,  0.3945,  0.1668),
    1.20: (-1.2347,  0.1302,  -0.5182,  0.3618,  0.1562),
    1.40: (-1.2436,  0.1249,  -0.5071,  0.3298,  0.1432),
    1.60: (-1.2447,  0.1220,  -0.4979,  0.3013,  0.1296),
    1.80: (-1.2426,  0.1205,  -0.4906,  0.2769,  0.1161),
    2.00: (-1.2399,  0.1196,  -0.4847,  0.2563,  0.1034),
    2.50: (-1.2355,  0.1185,  -0.4763,  0.2201,  0.0806),
    3.00: (-1.2344,  0.1182,  -0.4724,  0.2013,  0.0647),
}

DISTANCES = sorted(H2_COEFFS.keys())



# ──────────────────────────────────────────
# H₂ 분자 해밀토니안 (STO-3G 기저, JW 변환)
# 결합 거리별로 계수가 달라짐
# ──────────────────────────────────────────

def get_h2_hamiltonian(distance: float) -> SparsePauliOp:

    """문헌 계수 기반 H₂ 2-qubit 해밀토니안"""

    dists = np.array(DISTANCES)

    if distance in H2_COEFFS:
        g0, g1, g2, g3, g4 = H2_COEFFS[distance]

    else:
        # 선형 보간
        i = int(np.searchsorted(dists, distance))
        i = int(np.clip(i, 1, len(dists) - 1))
        d0, d1 = dists[i-1], dists[i]
        t = (distance - d0) / (d1 - d0)
        c0 = np.array(H2_COEFFS[d0])
        c1 = np.array(H2_COEFFS[d1])
        g0, g1, g2, g3, g4 = c0 + t * (c1 - c0)


    H = SparsePauliOp.from_list([
        ("II", g0),
        ("ZI", g1),
        ("IZ", g2),
        ("ZZ", g3),
        ("XX", g4),
        ("YY", g4),
    ])

    return H


def run_vqe_at_distance(distance: float, maxiter: int = 200) -> dict:
    """단일 결합 거리에서 VQE 실행"""
    
    H = get_h2_hamiltonian(distance)

    ansatz = TwoLocal(
        num_qubits=2,
        rotation_blocks=["ry", "rz"],
        entanglement_blocks="cz",
        reps = REPS
    )

    optimizer = OPTIMIZER_MAP[OPTIMIZER_TYPE](maxiter=maxiter)
    # optimizer = COBYLA(maxiter=maxiter)
    estimator = StatevectorEstimator()

    optimizer_name = optimizer.__class__.__name__

    vqe = VQE(estimator, ansatz, optimizer)
    result = vqe.compute_minimum_eigenvalue(H)

    return {
        "distance": distance,
        "energy": result.eigenvalue.real,
        # "optimal_params": result.optimal_parameters,
        "num_iterations": result.optimizer_result.nfev,
    }


def plot_energy_curve(distances: list, energies: list, save_path: str):
    """H₂ 포텐셜 에너지 곡선 시각화"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0d1117')

    for ax in axes:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#c9d1d9')
        ax.xaxis.label.set_color('#c9d1d9')
        ax.yaxis.label.set_color('#c9d1d9')
        ax.title.set_color('#f0f6fc')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')

    
    # 최소 에너지 지점 표시
    min_idx = np.argmin(energies)
    eq_dist = distances[min_idx]
    eq_energy = energies[min_idx]


    # ── 그래프 1: 포텐셜 에너지 곡선 ──
    ax1 = axes[0]
    ax1.plot(distances, energies, 'o-',
             color='#58a6ff', linewidth=2,
             markersize=6, markerfacecolor='#f78166',
             markeredgecolor='#f78166', label='VQE Energy')


    ax1.axvline(x=distances[min_idx], color='#3fb950',
                linestyle='--', alpha=0.7, label=f'Equilibrium: {distances[min_idx]:.2f} Å')
    ax1.axhline(y=energies[min_idx], color='#3fb950',
                linestyle='--', alpha=0.4)
    ax1.scatter([distances[min_idx]], [energies[min_idx]],
                color='#3fb950', s=100, zorder=5)

    ax1.set_xlabel('Bond Distance (Å)', fontsize=12)
    ax1.set_ylabel('Energy (Hartree)', fontsize=12)
    ax1.set_title('H₂ Potential Energy Curve (VQE)', fontsize=13, fontweight='bold')
    ax1.legend(facecolor='#21262d', edgecolor='#30363d',
               labelcolor='#c9d1d9', fontsize=10)
    ax1.grid(True, alpha=0.2, color='#30363d')

    # 에너지 최솟값 텍스트 표시
    ax1.annotate(
        f'E_min = {energies[min_idx]:.4f} Ha\nd = {distances[min_idx]:.3f} Å',
        xy=(distances[min_idx], energies[min_idx]),
        xytext=(distances[min_idx] + 0.3, energies[min_idx] + 0.08),
        fontsize=9, color='#3fb950',
        arrowprops=dict(arrowstyle='->', color='#3fb950', lw=1.5),
    )



    # ── 그래프 2: 에너지 vs 거리 (log scale 결합 해리 표시) ──
    ax2 = axes[1]
    # 상대 에너지 (해리 에너지 기준)
    dissociation_energy = energies[-1]
    relative_energies = [e - dissociation_energy for e in energies]

    ax2.fill_between(distances, relative_energies,
                     alpha=0.3, color='#58a6ff')
    ax2.plot(distances, relative_energies, 'o-',
             color='#58a6ff', linewidth=2, markersize=6,
             markerfacecolor='#f78166', markeredgecolor='#f78166')

    ax2.axhline(y=0, color='#8b949e', linestyle='-', alpha=0.5,
                label='Dissociation limit')
    ax2.axhline(y=relative_energies[min_idx], color='#d29922',
                linestyle='--', alpha=0.7,
                label=f'Binding energy: {abs(relative_energies[min_idx]):.4f} Ha')

    ax2.set_xlabel('Bond Distance (Å)', fontsize=12)
    ax2.set_ylabel('Relative Energy (Hartree)', fontsize=12)
    ax2.set_title('H₂ Binding Energy Analysis', fontsize=13, fontweight='bold')
    ax2.legend(facecolor='#21262d', edgecolor='#30363d',
               labelcolor='#c9d1d9', fontsize=10)
    ax2.grid(True, alpha=0.2, color='#30363d')

    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='#0d1117')
    plt.close()
    print(f"  그래프 저장 완료: {save_path}")



def generate_graph():

    np.random.seed(42)

    distances = DISTANCES
    # # 결합 거리 범위 (0.4 ~ 2.5 Å)
    # distances = [0.4, 0.5, 0.6, 0.7, 0.735, 0.8, 0.9,
    #              1.0, 1.2, 1.5, 1.8, 2.0, 2.5]

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("vqe_h2_energy_curve")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    energies = []
    all_results = []

    with mlflow.start_run(run_name="h2_potential_curve"):

        mlflow.log_param("molecule", "H2")
        mlflow.log_param("basis", "STO-3G (approximate)")
        mlflow.log_param("mapping", "Jordan-Wigner")
        mlflow.log_param("ansatz", "TwoLocal(ry,rz,cz,reps="+ str(REPS) +")")
        mlflow.log_param("optimizer", OPTIMIZER_TYPE)
        # mlflow.log_param("optimizer", "COBYLA")
        mlflow.log_param("num_distances", len(distances))

        print("=" * 50)
        print("  H₂ VQE Potential Energy Curve")
        print("=" * 50)

        for i, d in enumerate(distances):

            print(f"\n[{i+1}/{len(distances)}] 거리 = {d:.3f} Å 계산 중...")
            res = run_vqe_at_distance(d)
            energies.append(res["energy"])
            all_results.append(res)

            # MLflow에 각 거리별 에너지 기록
            mlflow.log_metric("energy", res["energy"], step=i)
            mlflow.log_metric("bond_distance", d, step=i)
            print(f"  에너지 = {res['energy']:.6f} Hartree  "
                  f"({res['num_iterations']} iterations)")
            

        # 최솟값 정보
        min_idx = np.argmin(energies)
        eq_distance = distances[min_idx]
        eq_energy = energies[min_idx]
        binding_energy = eq_energy - energies[-1]

        print("\n" + "=" * 50)
        print(f"  평형 거리:     {eq_distance:.3f} Å")
        print(f"  최소 에너지:   {eq_energy:.6f} Hartree")
        print(f"  결합 에너지:   {abs(binding_energy):.6f} Hartree")
        print("=" * 50)

        mlflow.log_metric("equilibrium_distance", eq_distance)
        mlflow.log_metric("minimum_energy", eq_energy)
        mlflow.log_metric("binding_energy", abs(binding_energy))

        # 그래프 생성
        graph_name = "h2_energy_curve_reps" + str(REPS) + "_" + OPTIMIZER_TYPE + ".png"
        json_name = "vqe_results_reps" + str(REPS) + "_"  + OPTIMIZER_TYPE + ".json"

        graph_path = str(results_dir / graph_name)

        plot_energy_curve(distances, energies, graph_path)
        mlflow.log_artifact(graph_path)

        # 결과 JSON 저장
        json_path = str(results_dir / json_name)
        with open(json_path, "w") as f:
            json.dump({
                "molecule": "H2",
                "equilibrium_distance_angstrom": eq_distance,
                "minimum_energy_hartree": eq_energy,
                "binding_energy_hartree": abs(binding_energy),
                "distances": distances,
                "energies": energies,
            }, f, indent=2)
        mlflow.log_artifact(json_path)

    print("\nMLflow 실험 기록 완료!")
    print("그래프: results/" + graph_name)



if __name__ == "__main__":
    generate_graph()
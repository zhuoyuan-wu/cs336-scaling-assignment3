import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

# 1. 加载数据
data_path = Path("data/isoflops_curves.json")
with open(data_path, "r") as f:
    data = json.load(f)

grouped = defaultdict(list)
for item in data:
    grouped[item["compute_budget"]].append(item)

budgets = sorted(grouped.keys())

# 2. 对每个预算执行对数空间二次抛物线拟合 (Quadratic Fit in log-space)
C_list = []
N_cont_list = []
D_cont_list = []
min_loss_list = []

print("=" * 80)
print(f"{'Budget (FLOPs)':<16} | {'Discrete N*':<14} | {'Quadratic N*':<14} | {'Quadratic D*':<14} | {'c2 (Curvature)':<12}")
print("-" * 80)

plt.figure(figsize=(10, 6))

for C in budgets:
    runs = grouped[C]
    params = np.array([r["parameters"] for r in runs], dtype=np.float64)
    losses = np.array([r["final_loss"] for r in runs], dtype=np.float64)
    
    # 取离散最小值供对比
    discrete_best = min(runs, key=lambda x: x["final_loss"])["parameters"]
    
    log_N = np.log(params)
    
    # 拟合抛物线: Loss = c2 * (log_N)^2 + c1 * log_N + c0
    # 注意: np.polyfit 返回 [c2, c1, c0]
    c2, c1, c0 = np.polyfit(log_N, losses, deg=2)
    
    # 确保抛物线开口向上 (c2 > 0)
    assert c2 > 0, f"预算 {C} 的拟合抛物线不是下凸的 (c2 <= 0)"
    
    # 解析求极小值点: ln(N*) = -c1 / (2 * c2)
    log_N_opt = -c1 / (2.0 * c2)
    N_opt_cont = np.exp(log_N_opt)
    D_opt_cont = C / (6.0 * N_opt_cont)
    loss_min_cont = c2 * (log_N_opt**2) + c1 * log_N_opt + c0
    
    C_list.append(C)
    N_cont_list.append(N_opt_cont)
    D_cont_list.append(D_opt_cont)
    min_loss_list.append(loss_min_cont)
    
    print(f"{C:<16.2e} | {discrete_best:<14.2e} | {N_opt_cont:<14.2e} | {D_opt_cont:<14.2e} | {c2:<12.4f}")
    
    # 绘制该预算下的点与拟合曲线
    x_fine = np.linspace(min(log_N) - 0.2, max(log_N) + 0.2, 100)
    y_fine = c2 * (x_fine**2) + c1 * x_fine + c0
    plt.scatter(params, losses, alpha=0.6)
    plt.plot(np.exp(x_fine), y_fine, linestyle=":", alpha=0.7)
    plt.scatter([N_opt_cont], [loss_min_cont], color="red", marker="x", s=60)

plt.xscale("log")
plt.xlabel("Model Parameters ($N$)", fontsize=12)
plt.ylabel("Validation / Training Loss", fontsize=12)
plt.title("Chinchilla Approach 1: IsoFLOP Profiles & Quadratic Valleys", fontsize=13)
plt.grid(True, which="both", ls=":", alpha=0.5)
plt.tight_layout()
plt.savefig("plots/isoflop_quadratic_profiles.png", dpi=300)
plt.close()

# 3. 幂律拟合: 连续谷底极值点拟合
C_arr = np.array(C_list)
N_arr = np.array(N_cont_list)
D_arr = np.array(D_cont_list)
log_C = np.log(C_arr)

# N_opt = A * C^a
a_cont, log_A_cont = np.polyfit(log_C, np.log(N_arr), deg=1)
A_cont = np.exp(log_A_cont)

# D_opt = B * C^b
b_cont, log_B_cont = np.polyfit(log_C, np.log(D_arr), deg=1)
B_cont = np.exp(log_B_cont)

print("=" * 80)
print("【Chinchilla Approach 1 连续拟合结果 vs 离散结果】")
print(f"连续抛物线极值: N_opt(C) = {A_cont:.4e} * C^{a_cont:.4f}")
print(f"连续抛物线极值: D_opt(C) = {B_cont:.4e} * C^{b_cont:.4f}")
print(f"守恒检验: a + b = {a_cont + b_cont:.4f}")
print("=" * 80)

# 4. 外推预测
print("\n【高算力外推预测】")
for target_C in [1e23, 1e24]:
    pred_N = A_cont * (target_C ** a_cont)
    pred_D = B_cont * (target_C ** b_cont)
    print(f"Budget C = {target_C:.1e} FLOPs:")
    print(f"  - 最优参数量 N*: {pred_N:.4e} ({pred_N/1e9:.2f} B)")
    print(f"  - 最优数据量 D*: {pred_D:.4e} ({pred_D/1e9:.2f} B tokens)")
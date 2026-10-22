import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

# 1. 加载并解析数据
data_path = Path("data/isoflops_curves.json")
with open(data_path, "r") as f:
    data = json.load(f)

grouped = defaultdict(list)
for item in data:
    grouped[item["compute_budget"]].append(item)

budgets = sorted(grouped.keys())

# 2. 提取每个预算下的经验最小值点 (N_opt, D_opt)
C_list = []
N_opt_list = []
D_opt_list = []
loss_opt_list = []

print(f"{'Budget (FLOPs)':<18} | {'N_opt (Params)':<16} | {'D_opt (Tokens)':<16} | {'Min Loss':<10}")
print("-" * 68)

for C in budgets:
    runs = grouped[C]
    # 取 final_loss 最小的 run
    best_run = min(runs, key=lambda x: x["final_loss"])
    N_opt = best_run["parameters"]
    # 严格按照 C = 6 * N * D 关系反算 D
    D_opt = C / (6.0 * N_opt)
    
    C_list.append(C)
    N_opt_list.append(N_opt)
    D_opt_list.append(D_opt)
    loss_opt_list.append(best_run["final_loss"])
    
    print(f"{C:<18.2e} | {N_opt:<16.2e} | {D_opt:<16.2e} | {best_run['final_loss']:<10.4f}")

C_arr = np.array(C_list, dtype=np.float64)
N_arr = np.array(N_opt_list, dtype=np.float64)
D_arr = np.array(D_opt_list, dtype=np.float64)

# 3. 对数空间线性拟合 (Power-Law Fitting)
log_C = np.log(C_arr)

# 拟合 N_opt = A * C^a
a, log_A = np.polyfit(log_C, np.log(N_arr), deg=1)
A = np.exp(log_A)

# 拟合 D_opt = B * C^b
b, log_B = np.polyfit(log_C, np.log(D_arr), deg=1)
B = np.exp(log_B)

print("\n" + "="*50)
print("【幂律拟合结果】")
print(f"N_opt(C) = {A:.4e} * C^{a:.4f}")
print(f"D_opt(C) = {B:.4e} * C^{b:.4f}")
print(f"理论检验: a + b = {a + b:.4f} (理论预期约为 1.0)")
print("="*50)

# 4. 外推预测 10^23 和 10^24
target_budgets = [1e23, 1e24]
print("\n【外推预测】")
for target_C in target_budgets:
    pred_N = A * (target_C ** a)
    pred_D = B * (target_C ** b)
    # 也可以用严格物理约束 C / (6 * pred_N) 来对照
    pred_D_exact = target_C / (6.0 * pred_N)
    print(f"Compute Budget C = {target_C:.1e} FLOPs:")
    print(f"  - 预测最优参数量 N_opt: {pred_N:.4e} ({pred_N/1e9:.2f} B)")
    print(f"  - 预测最优数据量 D_opt: {pred_D:.4e} ({pred_D/1e9:.2f} B tokens)")
    print(f"  - 物理约束检验 D = C/(6N): {pred_D_exact:.4e} ({pred_D_exact/1e9:.2f} B tokens)")

# 5. 生成学术级曲线 (Deliverable Plots)
Path("plots").mkdir(exist_ok=True)
plot_C = np.logspace(np.log10(min(C_arr)), 24, 200)
plot_N = A * (plot_C ** a)
plot_D = B * (plot_C ** b)

# 图 (a): Model Size vs. Compute Budget
plt.figure(figsize=(7, 5))
plt.scatter(C_arr, N_arr, color="#1f77b4", label="Observed $N_{opt}$ (from IsoFLOPs)", zorder=5)
plt.plot(plot_C, plot_N, color="#ff7f0e", linestyle="--", label=f"Fit: $N = {A:.2e} \\cdot C^{{{a:.3f}}}$")
plt.scatter(target_budgets, [A * (c**a) for c in target_budgets], color="red", marker="*", s=150, zorder=6, label="Extrapolated ($10^{23}, 10^{24}$)")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Compute Budget $C$ (FLOPs)", fontsize=12)
plt.ylabel("Optimal Parameters $N_{opt}$", fontsize=12)
plt.title("Compute-Optimal Model Size Scaling Law", fontsize=13)
plt.grid(True, which="both", ls=":", alpha=0.6)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig("plots/scaling_law_parameters.png", dpi=300)
plt.close()

# 图 (b): Dataset Size vs. Compute Budget
plt.figure(figsize=(7, 5))
plt.scatter(C_arr, D_arr, color="#2ca02c", label="Observed $D_{opt}$ (from IsoFLOPs)", zorder=5)
plt.plot(plot_C, plot_D, color="#d62728", linestyle="--", label=f"Fit: $D = {B:.2e} \\cdot C^{{{b:.3f}}}$")
plt.scatter(target_budgets, [B * (c**b) for c in target_budgets], color="purple", marker="*", s=150, zorder=6, label="Extrapolated ($10^{23}, 10^{24}$)")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Compute Budget $C$ (FLOPs)", fontsize=12)
plt.ylabel("Optimal Dataset Size $D_{opt}$ (Tokens)", fontsize=12)
plt.title("Compute-Optimal Dataset Size Scaling Law", fontsize=13)
plt.grid(True, which="both", ls=":", alpha=0.6)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig("plots/scaling_law_dataset.png", dpi=300)
plt.close()

print("\n[*] 绘图完成，已保存至 plots/scaling_law_parameters.png 与 plots/scaling_law_dataset.png")
import json
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# 1. 加载并平铺全部 72 个真实实验点
data_path = Path("data/isoflops_curves.json")
with open(data_path, "r") as f:
    data = json.load(f)

N_all = np.array([item["parameters"] for item in data], dtype=np.float64)
C_all = np.array([item["compute_budget"] for item in data], dtype=np.float64)
D_all = C_all / (6.0 * N_all)
L_all = np.array([item["final_loss"] for item in data], dtype=np.float64)

# 2. 定义 Huber Loss
def huber_loss(residual, delta=1e-3):
    abs_r = np.abs(residual)
    quadratic = np.minimum(abs_r, delta)
    linear = abs_r - quadratic
    return 0.5 * (quadratic ** 2) + delta * linear

# 3. 目标函数: 在对数预测空间最小化 Huber 损失
# 待优化向量: p = [ln(E), ln(A), ln(B), ln(alpha), ln(beta)]
def objective(p):
    E, A, B, alpha, beta = np.exp(p)
    
    # 预测损失
    L_pred = E + (A / (N_all ** alpha)) + (B / (D_all ** beta))
    
    # 计算在对数空间下的残差: ln(L_pred) - ln(L_all)
    # 优化是相对误差
    log_residuals = np.log(L_pred) - np.log(L_all)
    
    return np.sum(huber_loss(log_residuals, delta=1e-3))

# 4. 挑选物理合理的初始值 
# E 约在 1.5 ~ 3.5 之间
# alpha, beta 约在 0.2 ~ 0.5 之间
# A, B 约在 100 ~ 1000 之间

p0 = np.log([2.5, 400.0, 400.0, 0.35, 0.35])

# 5. 执行 L-BFGS 优化
res = minimize(objective, p0, method="L-BFGS-B", options={"maxiter": 2000, "ftol": 1e-12})

# 还原物理参数
E_opt, A_opt, B_opt, alpha_opt, beta_opt = np.exp(res.x)

# 6. 从全局曲面参数解析解出 Scaling 指数 a 和 b
# 理论公式: a = beta / (alpha + beta), b = alpha / (alpha + beta)
a_param = beta_opt / (alpha_opt + beta_opt)
b_param = alpha_opt / (alpha_opt + beta_opt)

# 最优系数 G (使得 N* = G * C^a)
# N* = [ (alpha * A) / (beta * B * 6^-beta) ]^(1 / (alpha + beta)) * C^(beta / (alpha + beta))
factor = (alpha_opt * A_opt) / (beta_opt * B_opt * (6.0 ** (beta_opt)))
G_param = factor ** (1.0 / (alpha_opt + beta_opt))
H_param = 1.0 / (6.0 * G_param)

print("=" * 60)
print("【Chinchilla Approach 3 全局曲面参数拟合结果】")
print(f"优化收敛状态: {res.success} ({res.message})")
print(f"  - E (不可约熵 Loss下界) : {E_opt:.4f}")
print(f"  - A (参数容量缩放因子)  : {A_opt:.4f}")
print(f"  - B (数据量缩放因子)    : {B_opt:.4f}")
print(f"  - alpha (参数幂律指数) : {alpha_opt:.4f}")
print(f"  - beta  (数据幂律指数) : {beta_opt:.4f}")
print("-" * 60)
print("【推导得出的最优扩张指数 (Optimal Exponents)】")
print(f"  - 最优模型扩张指数 a = beta / (alpha + beta) : {a_param:.4f}")
print(f"  - 最优数据扩张指数 b = alpha / (alpha + beta): {b_param:.4f}")
print(f"  - 物理守恒检验 a + b                       : {a_param + b_param:.4f}")
print(f"  - 闭式幂律公式: N_opt(C) = {G_param:.4e} * C^{a_param:.4f}")
print("=" * 60)

# 7. 外推预测对比
print("\n【高算力外推预测 (Approach 3 全局拟合 vs 目标)】")
for target_C in [1e23, 1e24]:
    pred_N = G_param * (target_C ** a_param)
    pred_D = target_C / (6.0 * pred_N)
    pred_L = E_opt + (A_opt / (pred_N ** alpha_opt)) + (B_opt / (pred_D ** beta_opt))
    print(f"Compute Budget C = {target_C:.1e} FLOPs:")
    print(f"  - 预测最优参数量 N*: {pred_N:.4e} ({pred_N/1e9:.2f} B)")
    print(f"  - 预测最优数据量 D*: {pred_D:.4e} ({pred_D/1e9:.2f} B tokens)")
    print(f"  - 预测理论极限 Loss: {pred_L:.4f}")
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

data_path = Path("data/isoflops_curves.json")
assert data_path.exists(), f"找不到文件: {data_path}"

with open(data_path, "r") as f:
    data = json.load(f)

print(f"[*] 样本总数: {len(data)}")
print(f"[*] 第一个样本结构: {data[0]}")

# 按 compute_budget 分组统计
grouped = defaultdict(list)
for item in data:
    grouped[item["compute_budget"]].append((item["parameters"], item["final_loss"]))

budgets = sorted(grouped.keys())
print(f"[*] 包含的 Compute Budgets 数量: {len(budgets)}")
print(f"[*] 算力预算列表 (FLOPs): {[f'{b:.1e}' for b in budgets]}")

print("\n[*] 各算力等级下的实验点分布:")
for b in budgets:
    runs = grouped[b]
    params = [r[0] for r in runs]
    losses = [r[1] for r in runs]
    print(f"  - Budget {b:.2e}: {len(runs)} 个点 | 参数跨度: [{min(params):.2e}, {max(params):.2e}] | Loss跨度: [{min(losses):.4f}, {max(losses):.4f}]")
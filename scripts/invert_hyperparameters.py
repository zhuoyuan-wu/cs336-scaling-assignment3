import numpy as np

def design_optimal_transformer(target_params, head_dim=128):
    """
    根据目标参数量 N*，利用 12 * n_layer * d_model^2 逆向推导最符合大模型工程规范的架构。
    约束:
    1. d_model 必须能被 head_dim 整除 (得到整数 n_heads)
    2. d_model 必须是 128 或 256 的倍数 (以对齐 GPU Tensor Core 访存)
    3. Aspect Ratio: d_model / n_layer 黄金比例在 60 ~ 120 之间
    """
    best_config = None
    min_diff = float("inf")
    
    # 搜索合理的层数范围 
    for n_layer in range(32, 128, 2):
        # 理论需要的 d_model = sqrt(N / (12 * n_layer))
        d_ideal = np.sqrt(target_params / (12.0 * n_layer))
        
        # 将 d_ideal 向上或向下对齐到 head_dim 的整数倍
        for factor in [-1, 0, 1]:
            n_heads = max(1, int(round(d_ideal / head_dim)) + factor)
            d_model = n_heads * head_dim
            
            # 计算实际参数量
            actual_params = 12 * n_layer * (d_model ** 2)
            diff = abs(actual_params - target_params)
            
            # 检查纵横比
            aspect_ratio = d_model / n_layer
            if 60 <= aspect_ratio <= 120:
                if diff < min_diff:
                    min_diff = diff
                    best_config = {
                        "n_layer": n_layer,
                        "d_model": d_model,
                        "n_heads": n_heads,
                        "head_dim": head_dim,
                        "d_ffn": int(8/3 * d_model) if (int(8/3 * d_model) % 256 == 0) else round(8/3 * d_model / 256) * 256,
                        "actual_non_embed_params": actual_params,
                        "aspect_ratio": aspect_ratio,
                        "target_diff_pct": (diff / target_params) * 100
                    }
                    
    return best_config

# 针对 10^23 FLOPs 的 N* = 9.114e10
c_1e23_params = 9.1144e10
config_1e23 = design_optimal_transformer(c_1e23_params)

# 针对 10^24 FLOPs 的 N* = 2.981e11
c_1e24_params = 2.9806e11
config_1e24 = design_optimal_transformer(c_1e24_params)

print("=" * 60)
print("【10^23 FLOPs 最优架构推荐配置】")
for k, v in config_1e23.items():
    print(f"  {k:<25}: {v}")
print("=" * 60)
print("【10^24 FLOPs 最优架构推荐配置】")
for k, v in config_1e24.items():
    print(f"  {k:<25}: {v}")
print("=" * 60)
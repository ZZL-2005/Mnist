import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

# 加载结果数据
with open("mlp_hidden_dim_experiment.pkl", "rb") as f:
    all_results = pickle.load(f)

hidden_dims = list(all_results.keys())
print(f"加载了 {len(hidden_dims)} 个隐藏层维度的结果: {hidden_dims}")

def plot_with_gradient_colors(colormap_name='viridis'):
    """使用指定的渐变色绘制曲线"""
    
    # 设置样式
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 获取渐变色
    colors = plt.get_cmap(colormap_name)(np.linspace(0, 1, len(hidden_dims)))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 绘制准确率曲线
    for i, h in enumerate(hidden_dims):
        ax1.plot(all_results[h]["acc"], label=f"Hidden {h}", 
                color=colors[i], linewidth=3, alpha=0.8)
    
    ax1.set_title("Test Accuracy vs Epochs", fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel("Epoch", fontsize=14)
    ax1.set_ylabel("Accuracy", fontsize=14)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.tick_params(labelsize=12)
    ax1.set_ylim(0, 1)
    
    # 绘制损失曲线
    for i, h in enumerate(hidden_dims):
        ax2.plot(all_results[h]["loss"], label=f"Hidden {h}", 
                color=colors[i], linewidth=3, alpha=0.8)
    
    ax2.set_title("Test Loss vs Epochs", fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel("Epoch", fontsize=14)
    ax2.set_ylabel("Loss", fontsize=14)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.tick_params(labelsize=12)
    
    # 添加统一的图例，放在图外右侧
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), 
              fontsize=11, title='Hidden Dims', title_fontsize=12, 
              frameon=True, fancybox=True, shadow=True)
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(right=0.82)
    
    # 保存图片
    plt.savefig(f"mlp_curves_{colormap_name}.png", dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.show()

# 生成几种不同的渐变色方案
print("生成不同渐变色方案的曲线图...")

# 方案1: viridis (蓝绿渐变)
print("1. Viridis (蓝→绿→黄)")
plot_with_gradient_colors('viridis')

# 方案2: plasma (紫红渐变)
print("2. Plasma (紫→红→黄)")
plot_with_gradient_colors('plasma')

# 方案3: coolwarm (蓝红渐变)
print("3. Coolwarm (蓝→白→红)")
plot_with_gradient_colors('coolwarm')

# 方案4: spectral (彩虹渐变)
print("4. Spectral (彩虹色)")
plot_with_gradient_colors('Spectral')

print("所有曲线图生成完成!")

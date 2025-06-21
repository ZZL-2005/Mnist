import os
# 解决OpenMP冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from tqdm import tqdm

# 加载训练结果
print("加载训练结果...")
with open(r"F:\手写数字识别\code\mlp_hidden_dim_experiment.pkl", "rb") as f:
    all_results = pickle.load(f)

print(f"加载完成! 包含 {len(all_results)} 个隐藏层维度的结果")

# 分析 epoch 点
analyze_epochs = [0, 1, 3, 6, 10, 13, 20, 30, 60]

# t-SNE计算函数
def compute_tsne_for_features(feature_snapshots, sample_size=800):
    """
    对保存的特征进行t-SNE计算
    sample_size: 每个epoch采样的数据点数量，减少计算时间
    """
    tsne_results = {}
    
    for epoch, (feats, lbls) in feature_snapshots.items():
        print(f"  计算epoch {epoch}的t-SNE (特征维度: {feats.shape[1]})...")
        
        # 采样数据以加速t-SNE计算
        if len(feats) > sample_size:
            indices = np.random.choice(len(feats), sample_size, replace=False)
            feats_sample = feats[indices]
            lbls_sample = lbls[indices]
        else:
            feats_sample = feats
            lbls_sample = lbls
        
        # 只有当特征维度>=2时才进行t-SNE
        if feats_sample.shape[1] >= 2:
            perplexity = min(30, len(feats_sample)-1)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, 
                       n_iter=300, learning_rate=200)
            tsne_result = tsne.fit_transform(feats_sample)
            tsne_results[epoch] = (tsne_result, lbls_sample)
        else:
            # 对于1维特征，直接使用原始特征作为x坐标，y坐标设为0
            tsne_result = np.column_stack([feats_sample.flatten(), np.zeros(len(feats_sample))])
            tsne_results[epoch] = (tsne_result, lbls_sample)
    
    return tsne_results

# 计算t-SNE（如果还没有的话）
print("\n检查并计算t-SNE...")
for h in tqdm(all_results.keys(), desc="计算t-SNE"):
    if "tsne" not in all_results[h] and "features" in all_results[h]:
        print(f"\n计算隐藏层维度 {h} 的t-SNE...")
        tsne_results = compute_tsne_for_features(all_results[h]["features"], sample_size=800)
        all_results[h]["tsne"] = tsne_results

print("t-SNE计算完成!")

# 可视化函数
def plot_tsne_evolution(hidden_dim, tsne_results):
    """绘制特定隐藏层维度的t-SNE演化 - 9宫格布局，统一坐标轴，正方形显示"""
    available_epochs = sorted([e for e in tsne_results.keys() if e in analyze_epochs])
    
    if not available_epochs:
        print(f"隐藏层维度 {hidden_dim} 没有可用的t-SNE数据")
        return

    rows, cols = 3, 3
    fig_size = 12
    fig, axes = plt.subplots(rows, cols, figsize=(fig_size, fig_size))
    axes = axes.flatten()
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # 获取全局坐标轴范围
    all_coords = [tsne_results[e][0] for e in available_epochs if e in tsne_results]
    all_coords = np.vstack(all_coords)
    x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
    y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05

    for i, epoch in enumerate(analyze_epochs):
        ax = axes[i]
        if epoch in available_epochs:
            tsne_coords, labels = tsne_results[epoch]
            for digit in range(10):
                mask = labels == digit
                if np.any(mask):
                    ax.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1], 
                               c=[colors[digit]], alpha=0.8, s=20, label=f'{digit}')
            ax.set_title(f'Epoch {epoch}', fontsize=14, pad=15, weight='bold')

            # 👇 特殊处理：Epoch 0 用自己的坐标范围
            if epoch == 0:
                x0_min, x0_max = tsne_coords[:, 0].min(), tsne_coords[:, 0].max()
                y0_min, y0_max = tsne_coords[:, 1].min(), tsne_coords[:, 1].max()
                xm = (x0_max - x0_min) * 0.2
                ym = (y0_max - y0_min) * 0.2
                ax.set_xlim(x0_min - xm, x0_max + xm)
                ax.set_ylim(y0_min - ym, y0_max + ym)
            else:
                ax.set_xlim(x_min - x_margin, x_max + x_margin)
                ax.set_ylim(y_min - y_margin, y_max + y_margin)


    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], 
                          markersize=10, label=f'{i}') for i in range(10)]
    fig.legend(handles, [str(i) for i in range(10)], 
               loc='center left', bbox_to_anchor=(1.05, 0.5), 
               title='Digits', fontsize=12, title_fontsize=14)

    plt.suptitle(f'Hidden Dimension {hidden_dim} - t-SNE Evolution', 
                 fontsize=18, y=0.96, weight='bold')
    fig.subplots_adjust(left=0.05, right=0.88, top=0.92, bottom=0.05, wspace=0.3, hspace=0.3)
    plt.savefig(f'tsne_evolution_hidden_{hidden_dim}.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
def plot_same_epoch_comparison(selected_dims, epoch):
    """比较不同隐藏层维度在同一epoch的t-SNE结果，统一坐标轴范围"""
    available_dims = [h for h in selected_dims if h in all_results and "tsne" in all_results[h] 
                      and epoch in all_results[h]["tsne"]]
    if not available_dims:
        print(f"没有隐藏层维度在epoch {epoch}有可用的t-SNE数据")
        return

    n_dims = len(available_dims)
    rows = (n_dims + 3) // 4
    cols = min(4, n_dims)

    fig_width = cols * 3
    fig_height = rows * 3
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # 获取全局坐标轴范围
    all_coords = []
    for h in available_dims:
        coords, _ = all_results[h]["tsne"][epoch]
        all_coords.append(coords)
    all_coords = np.vstack(all_coords)
    x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
    y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05

    for i, hidden_dim in enumerate(available_dims):
        if i >= len(axes): break
        ax = axes[i]
        tsne_coords, labels = all_results[hidden_dim]["tsne"][epoch]
        for digit in range(10):
            mask = labels == digit
            if np.any(mask):
                ax.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1], 
                           c=[colors[digit]], alpha=0.8, s=18)
        ax.set_title(f'Hidden {hidden_dim}', fontsize=13, pad=12, weight='bold')
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('black')

    for j in range(len(available_dims), len(axes)):
        axes[j].set_visible(False)

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], 
                          markersize=10, label=f'{i}') for i in range(10)]
    fig.legend(handles, [str(i) for i in range(10)], 
               loc='center left', bbox_to_anchor=(1.05, 0.5), 
               title='Digits', fontsize=12, title_fontsize=14)

    plt.suptitle(f'Hidden Dimensions Comparison - Epoch {epoch}', 
                 fontsize=17, y=0.96, weight='bold')
    fig.subplots_adjust(left=0.05, right=0.88, top=0.92, bottom=0.05, wspace=0.3, hspace=0.3)
    plt.savefig(f'tsne_comparison_epoch_{epoch}.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()


# 主要可视化
print("\n开始生成t-SNE可视化...")

# 1. 选择几个代表性的隐藏层维度显示演化过程
representative_dims = [1, 2, 4, 8, 16, 32, 64, 128, 256]
available_dims = [h for h in representative_dims if h in all_results and "tsne" in all_results[h]]

print(f"\n可用的隐藏层维度: {available_dims}")

# 为每个维度绘制演化过程
for hidden_dim in available_dims:  # 为所有可用维度生成演化图
    print(f"\n绘制隐藏层维度 {hidden_dim} 的t-SNE演化...")
    plot_tsne_evolution(hidden_dim, all_results[hidden_dim]["tsne"])

# 2. 比较不同隐藏层维度在最终epoch的表现
print(f"\n绘制不同隐藏层维度在epoch 60的比较...")
comparison_dims = [1, 2, 4, 8, 16, 32, 64, 128]
plot_same_epoch_comparison(comparison_dims, epoch=60)

# 3. 比较早期epoch的表现
print(f"\n绘制不同隐藏层维度在epoch 1的比较...")
plot_same_epoch_comparison(comparison_dims, epoch=1)

print("\n所有t-SNE可视化完成!")
print("生成的图片文件:")
print("- tsne_evolution_hidden_X.png: 各隐藏层维度的演化过程")
print("- tsne_comparison_epoch_X.png: 不同隐藏层维度的比较")

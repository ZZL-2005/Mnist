
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from tqdm import tqdm

# 加载数据
data = torch.load(r"subset_data\mnist_fixed_subset.pth", weights_only=False)
train_images, train_labels = data["train_images"], data["train_labels"]
test_images, test_labels = data["test_images"], data["test_labels"]

# 构建 DataLoader
batch_size = 512
train_dataset = TensorDataset(train_images, train_labels)
test_dataset = TensorDataset(test_images, test_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 分析 epoch 点
analyze_epochs = [0, 1, 3, 6, 10, 13, 20, 30, 60]

# MLP 模型定义
class MLP(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = self.flatten(x)
        features = self.relu(self.fc1(x))
        out = self.fc2(features)
        return out, features

# 单组训练函数（只做训练，不计算t-SNE）
def train_mlp(hidden_dim):
    model = MLP(hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    acc_list, loss_list = [], []
    feature_snapshots = {}  # 保存特定epoch的特征快照

    # 添加epoch进度条
    for epoch in tqdm(range(81), desc=f"Hidden {hidden_dim}", ncols=100):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total_loss = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits, features = model(x)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total_loss += criterion(logits, y).item()

        acc = correct / len(test_dataset)
        avg_loss = total_loss / len(test_loader)
        acc_list.append(acc)
        loss_list.append(avg_loss)

        # 每10个epoch输出一次进度
        if (epoch + 1) % 10 == 0:
            tqdm.write(f"Epoch {epoch+1:2d}: Acc={acc:.4f}, Loss={avg_loss:.4f}")

        # 保存特定epoch的特征用于后续t-SNE
        if epoch in analyze_epochs:
            model.eval()
            all_features, all_labels = [], []
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    logits, features = model(x)
                    all_features.append(features.cpu())
                    all_labels.append(y.cpu())
            
            feats = torch.cat(all_features).numpy()
            lbls = torch.cat(all_labels).numpy()
            feature_snapshots[epoch] = (feats, lbls)

    return acc_list, loss_list, feature_snapshots

# t-SNE计算函数（使用采样数据）
def compute_tsne_for_features(feature_snapshots, sample_size=1000):
    """
    对保存的特征进行t-SNE计算
    sample_size: 每个epoch采样的数据点数量，减少计算时间
    """
    tsne_results = {}
    
    for epoch, (feats, lbls) in feature_snapshots.items():
        print(f"计算epoch {epoch}的t-SNE (特征维度: {feats.shape[1]})...")
        
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
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(feats_sample)-1))
            tsne_result = tsne.fit_transform(feats_sample)
            tsne_results[epoch] = (tsne_result, lbls_sample)
        else:
            # 对于1维特征，直接使用原始特征作为x坐标，y坐标设为0
            tsne_result = np.column_stack([feats_sample.flatten(), np.zeros(len(feats_sample))])
            tsne_results[epoch] = (tsne_result, lbls_sample)
    
    return tsne_results
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_dims = [1,2,3,4,5,6,7,8,9,10,12,14,16, 32, 64, 128, 256]
all_results = {}

print(f"使用设备: {device}")
print(f"总共需要训练 {len(hidden_dims)} 个不同的隐藏层维度")
print("=" * 60)

# 添加总体进度条
for i, h in enumerate(tqdm(hidden_dims, desc="总进度", position=0)):
    print(f"\n[{i+1}/{len(hidden_dims)}] 开始训练隐藏层维度 = {h}")
    accs, losses, features = train_mlp(h)
    all_results[h] = {
        "acc": accs,
        "loss": losses,
        "features": features  # 保存特征快照
    }
    print(f"隐藏层维度 {h} 训练完成! 最终准确率: {accs[-1]:.4f}")

print("\n" + "=" * 60)
print("所有训练完成!")

# # 计算t-SNE（可选，训练完后单独执行）
# print("\n开始计算t-SNE可视化...")
# for h in tqdm(hidden_dims, desc="计算t-SNE"):
#     if h in all_results and "features" in all_results[h]:
#         tsne_results = compute_tsne_for_features(all_results[h]["features"], sample_size=500)
#         all_results[h]["tsne"] = tsne_results

# print("t-SNE计算完成!")

# print("\n" + "=" * 60)
# print("所有训练完成!")

# 保存结果
with open("mlp_hidden_dim_experiment.pkl", "wb") as f:
    pickle.dump(all_results, f)

# 绘制准确率与损失曲线
import matplotlib.cm as cm
import seaborn as sns

# 设置更美观的颜色风格
plt.style.use('seaborn-v0_8-whitegrid')

# 使用渐变色调色板，从蓝色到红色的渐变
colors = plt.cm.viridis(np.linspace(0, 1, len(hidden_dims)))  # viridis渐变色

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 绘制准确率曲线
for i, h in enumerate(hidden_dims):
    ax1.plot(all_results[h]["acc"], label=f"Hidden {h}", 
             color=colors[i], linewidth=2.8, alpha=0.85)

ax1.set_title("Test Accuracy vs Epochs", fontsize=16, fontweight='bold', pad=20)
ax1.set_xlabel("Epoch", fontsize=14)
ax1.set_ylabel("Accuracy", fontsize=14)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.tick_params(labelsize=12)
ax1.set_ylim(0, 1)

# 绘制损失曲线
for i, h in enumerate(hidden_dims):
    ax2.plot(all_results[h]["loss"], label=f"Hidden {h}", 
             color=colors[i], linewidth=2.8, alpha=0.85)

ax2.set_title("Test Loss vs Epochs", fontsize=16, fontweight='bold', pad=20)
ax2.set_xlabel("Epoch", fontsize=14)
ax2.set_ylabel("Loss", fontsize=14)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.tick_params(labelsize=12)

# 添加统一的图例，放在图外右侧
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), 
          fontsize=11, title='Hidden Dimensions', title_fontsize=13, 
          frameon=True, fancybox=True, shadow=True)

# 调整布局，为图例留出空间
plt.tight_layout()
plt.subplots_adjust(right=0.82)

# 保存高质量图片
plt.savefig("mlp_accuracy_loss_plot.png", dpi=300, bbox_inches='tight',
           facecolor='white', edgecolor='none')
plt.show()

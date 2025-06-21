import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载数据
data = torch.load("subset_data/mnist_fixed_subset.pth")
train_loader = DataLoader(TensorDataset(data['train_images'], data['train_labels']), batch_size=512, shuffle=True)
test_loader = DataLoader(TensorDataset(data['test_images'], data['test_labels']), batch_size=1000)
test_size = len(data['test_labels'])

# 模型定义
class MLP(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, hidden_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        feats = self.feature(x)
        out = self.classifier(feats)
        return out

# 单轮训练和评估函数
def train_model(hidden_dim, mode="normal", epochs=30):
    model = MLP(hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()

    acc_list, loss_list = [], []

    # 设置冻结模式
    if mode == "freeze_feature":
        # 冻结特征提取器，只训练分类器
        for param in model.feature.parameters():
            param.requires_grad = False
        print(f"  冻结特征提取器，只训练分类器")
    elif mode == "freeze_classifier":
        # 冻结分类器，只训练特征提取器
        for param in model.classifier.parameters():
            param.requires_grad = False
        print(f"  冻结分类器，只训练特征提取器")
    else:
        print(f"  正常训练模式")

    # 创建优化器，只优化需要梯度的参数
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    # 检查可训练参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  可训练参数: {trainable_params}/{total_params}")

    for epoch in tqdm(range(epochs), desc=f"Hidden {hidden_dim} [{mode}]", ncols=80):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total_loss = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total_loss += criterion(logits, y).item()

        acc = correct / test_size
        avg_loss = total_loss / len(test_loader)
        acc_list.append(acc)
        loss_list.append(avg_loss)

    print(f"  最终准确率: {acc_list[-1]:.4f}")
    return acc_list, loss_list

# 主实验逻辑
hidden_dims = [1, 8, 16, 32, 64, 128, 256]
all_results = {"normal": {}, "freeze_feature": {}, "freeze_classifier": {}}

for h in hidden_dims:
    print(f"\nTraining Hidden={h} [normal]")
    acc, loss = train_model(h, mode="normal")
    all_results["normal"][h] = (acc, loss)

    print(f"Training Hidden={h} [freeze_feature]")
    acc, loss = train_model(h, mode="freeze_feature")
    all_results["freeze_feature"][h] = (acc, loss)

    print(f"Training Hidden={h} [freeze_classifier]")
    acc, loss = train_model(h, mode="freeze_classifier")
    all_results["freeze_classifier"][h] = (acc, loss)

# 为每个隐藏层神经元个数单独画图，比较三种训练模式
plt.style.use('seaborn-v0_8-whitegrid')

# 定义三种模式的颜色和标签
mode_colors = {
    "normal": "#2E86AB",           # 蓝色
    "freeze_feature": "#A23B72",   # 紫红色
    "freeze_classifier": "#F18F01" # 橙色
}

mode_labels = {
    "normal": "Normal Training",
    "freeze_feature": "Freeze Feature Extractor", 
    "freeze_classifier": "Freeze Classifier"
}

print("\n生成对比图...")

for h in hidden_dims:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 绘制准确率曲线
    for mode in ["normal", "freeze_feature", "freeze_classifier"]:
        accs, losses = all_results[mode][h]
        ax1.plot(accs, label=mode_labels[mode], 
                color=mode_colors[mode], linewidth=3, alpha=0.8)
        ax2.plot(losses, label=mode_labels[mode], 
                color=mode_colors[mode], linewidth=3, alpha=0.8)
    
    # 设置准确率子图
    ax1.set_title(f"Hidden Dim {h}: Test Accuracy vs Epochs", 
                 fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel("Epoch", fontsize=14)
    ax1.set_ylabel("Accuracy", fontsize=14)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.tick_params(labelsize=12)
    ax1.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    
    # 设置损失子图
    ax2.set_title(f"Hidden Dim {h}: Test Loss vs Epochs", 
                 fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel("Epoch", fontsize=14)
    ax2.set_ylabel("Loss", fontsize=14)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.tick_params(labelsize=12)
    ax2.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(f"hidden_{h}_comparison.png", dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.show()
    
    print(f"✓ 已生成 Hidden {h} 的对比图")

print("\n所有对比图生成完成!")
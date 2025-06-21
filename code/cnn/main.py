
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from tqdm import tqdm
import os

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载数据
data = torch.load("subset_data/mnist_fixed_subset.pth")
train_loader = DataLoader(TensorDataset(data['train_images'], data['train_labels']), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(data['test_images'], data['test_labels']), batch_size=1000)

# CNN 模型定义
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=2)  # 输出 8x28x28
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)  # 输出 8x14x14

        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, padding=2)  # 输出 16x14x14
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)  # 输出 16x7x7

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 7 * 7, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        feat1 = x.detach().cpu()
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        feat2 = x.detach().cpu()
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x, feat1, feat2

# 训练函数
def train_cnn(epochs=10):
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out, _, _ = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 测试准确率
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out, _, _ = model(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
        print(f"Epoch {epoch+1}: Test Acc = {correct / len(data['test_labels']):.4f}")

    return model

# 可视化特征图
def visualize_feature_maps(model):
    model.eval()
    sample_img = data['test_images'][0].unsqueeze(0).to(device)  # 取一个样本
    with torch.no_grad():
        _, feat1, feat2 = model(sample_img)

    def plot_feature_map(feats, layer_name):
        grid_img = make_grid(feats.squeeze(0).unsqueeze(1), nrow=4, normalize=True, padding=1)
        npimg = grid_img.cpu().numpy()
        plt.figure(figsize=(6, 6))
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.title(f"Feature Maps of {layer_name}")
        plt.axis('off')
        plt.savefig(f"{layer_name}_feature_maps.png", dpi=300)
        plt.show()

    plot_feature_map(feat1, "Conv1")
    plot_feature_map(feat2, "Conv2")

# 主程序
model = train_cnn(epochs=10)
visualize_feature_maps(model)
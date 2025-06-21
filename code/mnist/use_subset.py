import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import sys
import os

# 添加路径以便导入subset.py中的函数
sys.path.append(os.path.dirname(__file__))
from subset import load_subset

# 加载保存的子集
train_images, train_labels, test_images, test_labels = load_subset('subset_data/mnist_fixed_subset.pth')

# 创建DataLoader
train_dataset = TensorDataset(train_images, train_labels)
test_dataset = TensorDataset(test_images, test_labels)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

# MLP模型定义
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        x = self.feature(x)
        return self.classifier(x)

# 训练函数
def train_model(model, optimizer, epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    train_acc_history, test_acc_history = [], []
    
    for epoch in range(1, epochs+1):
        # 训练
        model.train()
        correct = 0
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total_loss += loss.item()
        
        train_acc = correct / len(train_dataset)
        train_acc_history.append(train_acc)

        # 测试
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                correct += (logits.argmax(1) == y).sum().item()
        
        test_acc = correct / len(test_dataset)
        test_acc_history.append(test_acc)
        
        print(f"Epoch {epoch:2d}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
    
    return train_acc_history, test_acc_history

if __name__ == "__main__":
    # 训练模型
    print("使用固定子集训练MLP模型...\n")
    
    model = MLP()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_acc, test_acc = train_model(model, optimizer, epochs=15)
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_acc)+1), train_acc, 'o-', label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(test_acc)+1), test_acc, 's-', label='Test Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n最终结果:")
    print(f"训练准确率: {train_acc[-1]:.4f}")
    print(f"测试准确率: {test_acc[-1]:.4f}")

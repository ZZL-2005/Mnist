import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 1. 定义卷积神经网络
class CNNDigitClassifier(nn.Module):
    def __init__(self):
        super(CNNDigitClassifier, self).__init__()
        
        # 卷积层
        self.conv_layers = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28x28x32
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), # 28x28x32
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14x32
            nn.Dropout2d(0.25),
            
            # 第二个卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 14x14x64
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # 14x14x64
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7x64
            nn.Dropout2d(0.25),
        )
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# 2. 数据预处理和加载（减小数据集）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 下载并加载数据
full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
full_test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 减小数据集大小以加快训练
train_size = 15000  # 从60000减少到15000
test_size = 2500    # 从10000减少到2500

print(f"使用训练样本: {train_size}, 测试样本: {test_size}")

# 创建子集
train_indices = list(range(train_size))
test_indices = list(range(test_size))

train_dataset = Subset(full_train_dataset, train_indices)
test_dataset = Subset(full_test_dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

# 3. 初始化模型、损失函数和优化器
model = CNNDigitClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

print(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,}")

# 4. 训练函数
def train_model(model, train_loader, criterion, optimizer, scheduler, epochs=8):
    model.train()
    train_losses = []
    train_accuracies = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # 学习率调度
        scheduler.step()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        print(f'Epoch {epoch+1}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        
        # 每2个epoch测试一次
        if (epoch + 1) % 2 == 0:
            test_model(model, test_loader, verbose=False)
    
    return train_losses, train_accuracies

# 5. 测试函数
def test_model(model, test_loader, verbose=True):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    
    if verbose:
        print(f'Final Test Results - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%')
    else:
        print(f'  Test Accuracy: {test_acc:.2f}%')
    
    return test_loss, test_acc

# 6. 开始训练
print("开始训练CNN模型...")
print("注意：使用了较小的数据集，训练时间约1-2分钟")
train_losses, train_accuracies = train_model(model, train_loader, criterion, optimizer, scheduler, epochs=8)

# 7. 测试模型
print("\\n最终测试结果：")
test_loss, test_acc = test_model(model, test_loader)

# 8. 保存模型
os.makedirs('./models', exist_ok=True)
torch.save(model.state_dict(), './models/cnn_digit_classifier.pth')
print("\\nCNN模型已保存到 './models/cnn_digit_classifier.pth'")

# 9. 绘制训练过程
plt.figure(figsize=(15, 5))

# 损失曲线
plt.subplot(1, 3, 1)
plt.plot(train_losses, 'b-', linewidth=2, label='Training Loss')
plt.title('CNN Training Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# 准确率曲线
plt.subplot(1, 3, 2)
plt.plot(train_accuracies, 'r-', linewidth=2, label='Training Accuracy')
plt.title('CNN Training Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True, alpha=0.3)

# 模型架构可视化
plt.subplot(1, 3, 3)
# 简单的架构图
layers = ['Input\\n28x28x1', 'Conv1\\n28x28x32', 'Conv2\\n14x14x32', 
          'Conv3\\n14x14x64', 'Conv4\\n7x7x64', 'FC\\n512', 'Output\\n10']
y_pos = np.arange(len(layers))
plt.barh(y_pos, [1]*len(layers), color=['lightblue', 'lightgreen', 'lightgreen', 
                                       'lightcoral', 'lightcoral', 'lightyellow', 'lightpink'])
plt.yticks(y_pos, layers)
plt.xlabel('Layer Depth')
plt.title('CNN Architecture', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig('./models/cnn_training_progress.png', dpi=300, bbox_inches='tight')
plt.show()

# 10. 显示一些测试样本的预测结果
def show_cnn_predictions(model, test_loader, num_samples=8):
    model.eval()
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            if i >= num_samples:
                break
                
            data, target = data.to(device), target.to(device)
            output = model(data[:num_samples])
            probabilities = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            
            for j in range(min(num_samples, len(data))):
                img = data[j].cpu().squeeze()
                true_label = target[j].cpu().item()
                pred_label = predicted[j].cpu().item()
                confidence = probabilities[j][pred_label].cpu().item()
                
                axes[j].imshow(img, cmap='gray')
                color = 'green' if true_label == pred_label else 'red'
                axes[j].set_title(f'True: {true_label}, Pred: {pred_label}\\nConf: {confidence:.2%}', 
                                color=color, fontsize=10)
                axes[j].axis('off')
            break
    
    plt.suptitle('CNN Model Predictions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./models/cnn_prediction_samples.png', dpi=300, bbox_inches='tight')
    plt.show()

print("\\n显示CNN预测样本...")
show_cnn_predictions(model, test_loader)

# 11. 模型对比信息
print("\\n" + "="*60)
print("🎯 CNN模型训练完成!")
print("="*60)
print(f"📊 最终测试准确率: {test_acc:.2f}%")
print(f"⚡ 训练轮次: 8 epochs")
print(f"📦 数据集大小: {train_size:,} 训练样本, {test_size:,} 测试样本")
print(f"🔧 模型参数: {sum(p.numel() for p in model.parameters()):,}")
print(f"💾 模型文件: ./models/cnn_digit_classifier.pth")
print("="*60)
print("现在可以运行交互式演示:")
print("python interactive_cnn_demo.py")
print("="*60)

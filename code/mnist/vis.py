import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# 加载MNIST数据集
transform = transforms.ToTensor()
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 获取训练集图像和标签
train_images = torch.stack([img for img, _ in mnist_train])
train_labels = torch.tensor([label for _, label in mnist_train])

# 获取测试集图像和标签  
test_images = torch.stack([img for img, _ in mnist_test])
test_labels = torch.tensor([label for _, label in mnist_test])

# 合并训练集和测试集
all_images = torch.cat([train_images, test_images], dim=0)
all_labels = torch.cat([train_labels, test_labels], dim=0)

# ---------- 图像统计 ----------
def plot_label_distribution_comparison(train_labels, test_labels):
    # 统计每个类别的数量
    train_counts = np.bincount(train_labels.numpy(), minlength=10)
    test_counts = np.bincount(test_labels.numpy(), minlength=10)
    
    # 创建DataFrame用于seaborn绘图
    data = []
    for i in range(10):
        data.append({'Digit': i, 'Count': train_counts[i], 'Dataset': 'Train'})
        data.append({'Digit': i, 'Count': test_counts[i], 'Dataset': 'Test'})
    
    df = pd.DataFrame(data)
    
    # 绘制并排柱状图
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Digit', y='Count', hue='Dataset', palette=['skyblue', 'lightcoral'])
    plt.title('MNIST Dataset Distribution: Train vs Test', fontsize=14)
    plt.xlabel('Digit Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend(title='Dataset')
    plt.tight_layout()
    plt.show()

# 所有图像的像素均值图（28x28）
def plot_pixel_mean_std(images):
    mean_image = images.mean(dim=0).squeeze()     # [28,28]
    std_image  = images.std(dim=0).squeeze()      # [28,28]

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(mean_image, ax=axs[0], cmap='viridis')
    axs[0].set_title("Mean Pixel Intensity")
    sns.heatmap(std_image, ax=axs[1], cmap='magma')
    axs[1].set_title("Pixel-wise Std Deviation")
    plt.tight_layout()
    plt.show()

# ---------- 可视化 ----------
print(f"训练集大小: {len(train_labels)}")
print(f"测试集大小: {len(test_labels)}")
print(f"总数据集大小: {len(all_labels)}")

plot_label_distribution_comparison(train_labels, test_labels)
plot_pixel_mean_std(all_images)  # 使用全部图像计算统计
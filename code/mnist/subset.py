import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
####绘图函数已被注释，可以自己重新打开
# 设定参数
train_per_class = 500
test_per_class = 300

# 加载 MNIST 原始数据集
transform = transforms.ToTensor()
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# ---------- 按顺序固定选取样本 ----------
def select_fixed_subset(dataset, per_class):
    class_indices = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(dataset):
        label = int(label)
        if len(class_indices[label]) < per_class:
            class_indices[label].append(idx)
        if all(len(idxs) == per_class for idxs in class_indices.values()):
            break
    selected_indices = [i for indices in class_indices.values() for i in indices]
    return Subset(dataset, selected_indices)

train_subset = select_fixed_subset(mnist_train, train_per_class)
test_subset  = select_fixed_subset(mnist_test, test_per_class)

# 提取图像和标签
def extract_images_labels(dataset):
    images = torch.stack([img for img, _ in dataset])
    labels = torch.tensor([label for _, label in dataset])
    return images, labels

train_images, train_labels = extract_images_labels(train_subset)
test_images, test_labels   = extract_images_labels(test_subset)
all_images = torch.cat([train_images, test_images], dim=0)
all_labels = torch.cat([train_labels, test_labels], dim=0)

# ---------- 图像统计可视化 ----------
def plot_label_distribution_comparison(train_labels, test_labels):
    train_counts = np.bincount(train_labels.numpy(), minlength=10)
    test_counts  = np.bincount(test_labels.numpy(), minlength=10)

    data = []
    for i in range(10):
        data.append({'Digit': i, 'Count': train_counts[i], 'Dataset': 'Train'})
        data.append({'Digit': i, 'Count': test_counts[i], 'Dataset': 'Test'})

    df = pd.DataFrame(data)

    # plt.figure(figsize=(10, 5))
    # sns.barplot(data=df, x='Digit', y='Count', hue='Dataset', palette=['skyblue', 'salmon'])
    # plt.title('MNIST Label Distribution (Fixed Subset)', fontsize=14)
    # plt.xlabel('Digit')
    # plt.ylabel('Sample Count')
    # plt.grid(True, axis='y', alpha=0.3)
    # plt.tight_layout()
    # plt.show()

def plot_pixel_mean_std(images):
    mean_image = images.mean(dim=0).squeeze()
    std_image  = images.std(dim=0).squeeze()

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(mean_image, ax=axs[0], cmap='viridis')
    axs[0].set_title("Mean Pixel Intensity")
    sns.heatmap(std_image, ax=axs[1], cmap='magma')
    axs[1].set_title("Pixel-wise Std Deviation")
    # plt.tight_layout()
    # plt.show()

# ---------- 保存子集 ----------
def save_subset(train_images, train_labels, test_images, test_labels, save_dir='subset_data'):
    """保存子集数据"""
    os.makedirs(save_dir, exist_ok=True)
    
    subset_data = {
        'train_images': train_images,
        'train_labels': train_labels,
        'test_images': test_images,
        'test_labels': test_labels,
        'train_per_class': train_per_class,
        'test_per_class': test_per_class,
        'total_train_size': len(train_labels),
        'total_test_size': len(test_labels)
    }
    
    save_path = os.path.join(save_dir, 'mnist_fixed_subset.pth')
    torch.save(subset_data, save_path)
    print(f"子集已保存到: {save_path}")
    return save_path

def load_subset(subset_path='subset_data/mnist_fixed_subset.pth'):
    """加载保存的子集数据"""
    if not os.path.exists(subset_path):
        print(f"子集文件不存在: {subset_path}")
        print("请先运行此脚本创建子集")
        return None
    
    data = torch.load(subset_path)
    print(f"从 {subset_path} 加载子集:")
    print(f"  训练集: {data['total_train_size']} 张 (每类 {data['train_per_class']} 张)")
    print(f"  测试集: {data['total_test_size']} 张 (每类 {data['test_per_class']} 张)")
    
    return data['train_images'], data['train_labels'], data['test_images'], data['test_labels']

# ---------- 运行 ----------
print(f"训练集子集大小: {len(train_labels)}")
print(f"测试集子集大小: {len(test_labels)}")
print(f"总子集大小: {len(all_labels)}")

# 保存子集
save_path = save_subset(train_images, train_labels, test_images, test_labels)

# 验证加载
print("\n验证加载功能:")
loaded_data = load_subset()

plot_label_distribution_comparison(train_labels, test_labels)
plot_pixel_mean_std(all_images)

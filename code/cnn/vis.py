import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# 添加网络结构可视化的库
try:
    from torchsummary import summary
    TORCHSUMMARY_AVAILABLE = True
except ImportError:
    TORCHSUMMARY_AVAILABLE = False
    print("torchsummary未安装，可以用: pip install torchsummary")

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

def print_network_structure(model):
    """打印详细的网络结构信息"""
    print("=" * 80)
    print("CNN 网络结构详情")
    print("=" * 80)
    
    # 1. 模型概览
    print("\n📋 模型概览:")
    print(model)
    
    # 2. 使用torchsummary
    if TORCHSUMMARY_AVAILABLE:
        print("\n📊 详细摘要:")
        try:
            summary(model, (1, 28, 28))
        except Exception as e:
            print(f"torchsummary出错: {e}")
    
    # 3. 参数统计
    print("\n🔢 参数统计:")
    total_params = 0
    trainable_params = 0
    
    layer_info = []
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
        layer_info.append({
            'name': name,
            'shape': list(param.shape),
            'params': param_count
        })
        print(f"  {name:15s}: {param_count:8,d} 参数, 形状: {list(param.shape)}")
    
    print(f"\n📈 总参数数量: {total_params:,}")
    print(f"🎯 可训练参数: {trainable_params:,}")
    
    # 4. 数据流分析
    print("\n🔄 数据流分析:")
    x = torch.randn(1, 1, 28, 28)
    print(f"  输入:        {list(x.shape):15s} -> 像素数: {x.numel():,}")
    
    x = model.conv1(x)
    print(f"  Conv1:       {list(x.shape):15s} -> 特征数: {x.numel():,}")
    x = model.relu1(x)
    print(f"  ReLU1:       {list(x.shape):15s} -> 特征数: {x.numel():,}")
    x = model.pool1(x)
    print(f"  MaxPool1:    {list(x.shape):15s} -> 特征数: {x.numel():,}")
    
    x = model.conv2(x)
    print(f"  Conv2:       {list(x.shape):15s} -> 特征数: {x.numel():,}")
    x = model.relu2(x)
    print(f"  ReLU2:       {list(x.shape):15s} -> 特征数: {x.numel():,}")
    x = model.pool2(x)
    print(f"  MaxPool2:    {list(x.shape):15s} -> 特征数: {x.numel():,}")
    
    x = model.flatten(x)
    print(f"  Flatten:     {list(x.shape):15s} -> 特征数: {x.numel():,}")
    x = model.fc(x)
    print(f"  全连接:       {list(x.shape):15s} -> 输出数: {x.numel():,}")

def draw_cnn_architecture():
    """绘制CNN架构图"""
    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    
    # 定义层信息
    layers = [
        {"name": "输入层", "detail": "1×28×28", "pos": (1, 4), "size": (1.5, 2), "color": "#E3F2FD"},
        {"name": "Conv1", "detail": "8×28×28\nK=5×5, P=2", "pos": (4, 4), "size": (1.5, 2), "color": "#C8E6C9"},
        {"name": "ReLU1", "detail": "激活函数", "pos": (6.5, 4), "size": (1, 1), "color": "#FFE0B2"},
        {"name": "MaxPool1", "detail": "8×14×14\n2×2池化", "pos": (9, 4), "size": (1.5, 1.5), "color": "#FFCDD2"},
        {"name": "Conv2", "detail": "16×14×14\nK=5×5, P=2", "pos": (12, 4), "size": (1.5, 1.5), "color": "#C8E6C9"},
        {"name": "ReLU2", "detail": "激活函数", "pos": (14.5, 4), "size": (1, 1), "color": "#FFE0B2"},
        {"name": "MaxPool2", "detail": "16×7×7\n2×2池化", "pos": (17, 4.25), "size": (1.5, 1), "color": "#FFCDD2"},
        {"name": "Flatten", "detail": "784维", "pos": (20, 4.5), "size": (1, 0.5), "color": "#F3E5F5"},
        {"name": "全连接", "detail": "10类输出", "pos": (23, 4.5), "size": (1.5, 0.5), "color": "#E1F5FE"},
    ]
    
    # 绘制各层
    for i, layer in enumerate(layers):
        x, y = layer["pos"]
        w, h = layer["size"]
        
        # 绘制矩形
        rect = patches.FancyBboxPatch(
            (x-w/2, y-h/2), w, h,
            boxstyle="round,pad=0.1",
            facecolor=layer["color"],
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(rect)
        
        # 添加层名称
        ax.text(x, y+0.2, layer["name"], ha='center', va='center', 
               fontsize=12, fontweight='bold')
        
        # 添加详细信息
        ax.text(x, y-0.2, layer["detail"], ha='center', va='center', 
               fontsize=10, style='italic')
        
        # 绘制箭头
        if i < len(layers) - 1:
            next_x = layers[i+1]["pos"][0] - layers[i+1]["size"][0]/2
            arrow_start = x + w/2
            arrow_end = next_x - 0.1
            
            ax.annotate('', xy=(arrow_end, y), xytext=(arrow_start, y),
                       arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))
    
    # 添加标题和说明
    ax.text(12.5, 7, 'CNN网络结构图', ha='center', va='center', 
           fontsize=18, fontweight='bold')
    
    # 添加参数信息
    param_text = """
    参数统计:
    • Conv1: 8×(1×5×5+1) = 208 参数
    • Conv2: 16×(8×5×5+1) = 3,216 参数  
    • FC: 784×10+10 = 7,850 参数
    • 总计: 11,274 参数
    """
    ax.text(1, 1.5, param_text, ha='left', va='top', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    ax.set_xlim(-1, 26)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("cnn_structure_detailed.png", dpi=300, bbox_inches='tight')
    plt.show()

def create_3d_visualization():
    """创建3D风格的网络结构图"""
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 定义层的3D位置和大小
    layers_3d = [
        {"name": "输入\n1×28×28", "pos": (0, 0, 0), "size": (28, 28, 1), "color": "lightblue"},
        {"name": "Conv1+ReLU\n8×28×28", "pos": (3, 0, 0), "size": (28, 28, 8), "color": "lightgreen"},
        {"name": "MaxPool1\n8×14×14", "pos": (6, 7, 0), "size": (14, 14, 8), "color": "lightcoral"},
        {"name": "Conv2+ReLU\n16×14×14", "pos": (9, 7, 0), "size": (14, 14, 16), "color": "lightgreen"},
        {"name": "MaxPool2\n16×7×7", "pos": (12, 10.5, 0), "size": (7, 7, 16), "color": "lightcoral"},
        {"name": "FC\n10", "pos": (15, 12, 0), "size": (1, 1, 10), "color": "lightyellow"},
    ]
    
    for layer in layers_3d:
        x, y, z = layer["pos"]
        dx, dy, dz = layer["size"]
        
        # 调整显示比例
        dx, dy, dz = dx/5, dy/5, dz/2
        
        # 绘制3D方块
        ax.bar3d(x, y, z, dx, dy, dz, color=layer["color"], alpha=0.7, edgecolor='black')
        
        # 添加标签
        ax.text(x+dx/2, y+dy/2, z+dz+1, layer["name"], ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('网络深度')
    ax.set_ylabel('空间维度')
    ax.set_zlabel('通道数')
    ax.set_title('CNN 3D 结构可视化', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("cnn_3d_structure.png", dpi=300, bbox_inches='tight')
    plt.show()

# 主程序
if __name__ == "__main__":
    # 创建模型
    model = SimpleCNN().to(device)
    
    # 1. 打印详细结构信息
    print_network_structure(model)
    
    # 2. 绘制2D架构图
    print("\n🎨 生成2D架构图...")
    draw_cnn_architecture()
    
    # 3. 绘制3D可视化
    print("\n🎨 生成3D结构图...")
    create_3d_visualization()
    
    print("\n✅ 网络结构展示完成！")
    print("📁 生成的文件:")
    print("   • cnn_structure_detailed.png - 详细2D架构图")
    print("   • cnn_3d_structure.png - 3D结构可视化")
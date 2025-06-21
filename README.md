# MNIST 手写数字识别项目

🎯 一个完整的手写数字识别项目，包含多种神经网络实现方案、详细的实验分析和可视化工具。

## 📁 项目结构

```
Mnist/
├── README.md                    # 项目说明文档
├── code/                        # 核心代码目录
│   ├── demo/                    # 演示和启动脚本
│   │   ├── run_project.py       # 🚀 项目主启动脚本
│   │   ├── interactive_demo.py  # MLP交互式演示
│   │   ├── interactive_cnn_demo.py # CNN交互式演示  
│   │   ├── train_digit_classifier.py # MLP训练脚本
│   │   └── train_cnn_classifier.py   # CNN训练脚本
│   ├── bp_by_hand/             # 手工实现反向传播
│   │   └── main.py             # 纯NumPy实现的MLP
│   ├── cnn/                    # 卷积神经网络
│   │   ├── main.py             # CNN主训练脚本
│   │   └── vis.py              # CNN特征可视化
│   ├── experiment2/            # 隐层维度实验
│   │   ├── main.py             # 多隐层维度对比实验
│   │   ├── tsne.py             # t-SNE降维可视化
│   │   └── plot_curves.py      # 实验结果绘图
│   ├── experiment3/            # 冻结实验
│   │   └── main.py             # 参数冻结对比实验
│   ├── mnist/                  # 数据处理工具
│   │   ├── subset.py           # 数据子集生成
│   │   ├── use_subset.py       # 子集使用工具
│   │   └── vis.py              # 数据可视化
│   ├── subset_data/            # 数据子集存储
│   │   ├── mnist_fixed_subset.pth     # PyTorch格式
│   │   └── mnist_fixed_subset_npz.npz # NumPy格式
│   └── data/                   # 原始MNIST数据
├── 报告/                       # 实验报告和论文
│   ├── latex/                  # LaTeX源文件
│   ├── images/                 # 报告图片资源
│   └── sections/               # 报告章节文件
├── 参考文献/                   # 相关参考资料
└── data/                       # 数据存储目录
```

## 🌟 主要特性

### 🧠 多种神经网络实现
- **手工BP神经网络**: 纯NumPy实现，深入理解反向传播原理
- **PyTorch MLP**: 现代深度学习框架实现的多层感知机
- **CNN卷积网络**: 针对图像识别优化的卷积神经网络

### 🔬 丰富的实验分析
- **隐层维度实验**: 对比不同隐层大小对性能的影响
- **特征可视化**: t-SNE降维展示特征学习过程
- **参数冻结实验**: 分析特征提取器和分类器的作用
- **训练过程可视化**: 详细的训练曲线和性能指标

### 🎮 交互式演示
- **实时绘图识别**: 支持鼠标绘制数字进行实时识别
- **模型对比演示**: 直观比较不同模型的识别效果
- **特征图可视化**: 展示CNN各层的特征激活情况

## 🚀 快速开始

### 环境要求
```bash
Python >= 3.7
PyTorch >= 1.8.0
torchvision
numpy
matplotlib
scikit-learn
Pillow
opencv-python
tqdm
seaborn
```

### 一键启动
```bash
cd code/demo
python run_project.py
```

### 分步运行

#### 1. 训练手工BP神经网络
```bash
cd code/bp_by_hand
python main.py
```

#### 2. 训练CNN模型
```bash
cd code/cnn
python main.py
```

#### 3. 运行隐层维度实验
```bash
cd code/experiment2
python main.py
```

#### 4. 启动交互式演示
```bash
cd code/demo
python interactive_cnn_demo.py  # CNN演示（推荐）
python interactive_demo.py      # MLP演示
```

## 📊 实验结果

### 模型性能对比
| 模型类型 | 测试准确率 | 训练时间 | 参数量 |
|---------|-----------|---------|--------|
| 手工BP-MLP | ~95.2% | 较长 | 23,540 |
| PyTorch-MLP | ~97.8% | 中等 | 可调 |
| Simple-CNN | ~99.1% | 较短 | 8,906 |

### 关键发现
- **CNN在图像识别任务上显著优于MLP**
- **隐层维度在16-128之间性能较稳定**
- **特征提取器比分类器对最终性能影响更大**
- **t-SNE显示深度网络能学习到更好的特征表示**

## 🔧 模块说明

### 核心算法模块

#### `bp_by_hand/main.py`
- 完全基于NumPy的手工实现
- 包含前向传播、反向传播、梯度更新全过程
- 适合学习深度学习基础原理

#### `cnn/main.py`
- 基于PyTorch的卷积神经网络
- 2层卷积+池化+全连接结构
- 包含特征图提取和可视化功能

#### `experiment2/main.py`
- 系统性隐层维度对比实验
- 自动化训练多个不同配置的模型
- 生成详细的性能对比图表

### 可视化工具

#### `experiment2/tsne.py`
- t-SNE降维可视化隐层特征
- 展示不同训练阶段的特征演化
- 支持多个隐层维度的对比分析

#### `cnn/vis.py`
- CNN特征图可视化
- 卷积核权重可视化
- 支持逐层特征激活展示

### 交互式工具

#### `demo/interactive_cnn_demo.py`
- 基于Tkinter的图形界面
- 支持鼠标绘制数字识别
- 实时显示CNN各层特征响应

## 📈 使用指南

### 数据准备
项目使用MNIST数据集的固定子集，确保实验结果可重现：
- 训练集: 6,000 样本
- 测试集: 1,000 样本
- 数据格式: 28×28灰度图像

### 自定义实验
1. **修改网络结构**: 编辑相应模型定义文件
2. **调整超参数**: 修改学习率、批次大小等参数
3. **扩展实验**: 基于现有框架添加新的实验分析

### 结果分析
- 训练曲线图保存在各模块目录下
- 混淆矩阵展示分类错误模式  
- t-SNE图展示特征学习效果
- 特征图显示CNN的感受野激活

## 🎯 项目亮点

1. **教学友好**: 从手工实现到现代框架，循序渐进
2. **实验完整**: 包含多角度的性能分析和可视化
3. **交互体验**: 提供直观的实时演示界面
4. **结果可重现**: 固定随机种子和数据划分
5. **文档详细**: 完整的技术报告和代码注释

## 📝 技术报告

详细的实验分析和理论推导请参阅 [`报告/latex/main.pdf`](报告/latex/main.pdf)

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目！

## 📄 许可证

本项目仅供学习和研究使用。

---
*如有问题或建议，请通过GitHub Issues联系作者*
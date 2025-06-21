#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
手写数字识别项目启动脚本
"""

import os
import sys
import subprocess

def check_requirements():
    """检查必要的库是否安装"""
    required_packages = [
        'torch', 'torchvision', 'numpy', 'matplotlib', 
        'Pillow', 'opencv-python', 'tqdm'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"缺少以下依赖包: {', '.join(missing_packages)}")
        install = input("是否自动安装？(y/n): ")
        if install.lower() == 'y':
            for package in missing_packages:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        else:
            print("请手动安装缺少的包后再运行")
            return False
    return True

def main():
    """主函数"""
    print("=" * 50)
    print("🎯 手写数字识别项目")
    print("=" * 50)
    
    if not check_requirements():
        return
    
    while True:
        print("\\n请选择操作:")
        print("1. 训练全连接神经网络 (MLP)")
        print("2. 训练卷积神经网络 (CNN) - 推荐")
        print("3. 运行MLP交互式演示")
        print("4. 运行CNN交互式演示 - 推荐")
        print("5. 退出")
        
        choice = input("\\n请输入选择 (1-5): ").strip()
        
        if choice == '1':
            print("\\n🚀 开始训练MLP模型...")
            print("这可能需要几分钟时间，请耐心等待...")
            os.system("python train_digit_classifier.py")
            
        elif choice == '2':
            print("\\n🧠 开始训练CNN模型...")
            print("使用较小数据集，训练时间约1-2分钟...")
            os.system("python train_cnn_classifier.py")
            
        elif choice == '3':
            if not os.path.exists('./models/digit_classifier.pth'):
                print("\\n❌ 未找到MLP模型！")
                print("请先选择选项1训练MLP模型")
                continue
                
            print("\\n🎨 启动MLP交互式演示...")
            os.system("python interactive_demo.py")
            
        elif choice == '4':
            if not os.path.exists('./models/cnn_digit_classifier.pth'):
                print("\\n❌ 未找到CNN模型！")
                print("请先选择选项2训练CNN模型")
                continue
                
            print("\\n🧠 启动CNN交互式演示...")
            os.system("python interactive_cnn_demo.py")
            
        elif choice == '5':
            print("\\n👋 再见！")
            break
            
        else:
            print("\\n❌ 无效选择，请重新输入")

if __name__ == "__main__":
    main()

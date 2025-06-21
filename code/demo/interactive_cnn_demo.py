import tkinter as tk
from tkinter import ttk, messagebox, font
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
# 设置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 定义CNN网络结构（必须与训练时一致）
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

class CNNDigitRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CNN手写数字识别演示")
        self.root.geometry("1000x700")
        self.root.configure(bg='#2c3e50')
        
        # 设置默认字体，确保中文显示正常
        self.default_font = ('Microsoft YaHei', 10)
        self.title_font = ('Microsoft YaHei', 22, 'bold')
        self.button_font = ('Microsoft YaHei', 12, 'bold')
        self.result_font = ('Microsoft YaHei', 56, 'bold')
        
        # 加载模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()
        
        # 画布设置
        self.canvas_size = 280
        self.brush_size = 20
        
        # 创建界面
        self.create_widgets()
        
        # 画布图像
        self.canvas_image = Image.new("RGB", (self.canvas_size, self.canvas_size), "white")
        self.canvas_draw = ImageDraw.Draw(self.canvas_image)
        
    def load_model(self):
        """加载训练好的CNN模型"""
        try:
            model = CNNDigitClassifier()
            model.load_state_dict(torch.load('./models/cnn_digit_classifier.pth', map_location=self.device))
            model.to(self.device)
            model.eval()
            print("CNN模型加载成功!")
            return model
        except Exception as e:
            messagebox.showerror("错误", f"无法加载CNN模型: {e}\\n请先运行 train_cnn_classifier.py 训练模型")
            return None
    
    def create_widgets(self):
        """创建GUI组件"""
        # 主框架
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        # 左侧框架（画布和控制）
        left_frame = tk.Frame(main_frame, bg='#2c3e50')
        left_frame.pack(side='left', fill='both', expand=True)
          # 标题
        title_label = tk.Label(left_frame, text="CNN手写数字识别", 
                              font=self.title_font, 
                              bg='#2c3e50', fg='#ecf0f1')
        title_label.pack(pady=(0, 20))
        
        # 模型信息
        model_info = tk.Label(left_frame, text="基于卷积神经网络 | GPU加速", 
                             font=self.default_font, 
                             bg='#2c3e50', fg='#95a5a6')
        model_info.pack(pady=(0, 15))
        
        # 画布框架
        canvas_frame = tk.Frame(left_frame, relief='raised', bd=3, bg='#34495e')
        canvas_frame.pack(pady=10)
        
        # 绘图画布
        self.canvas = tk.Canvas(canvas_frame, 
                               width=self.canvas_size, 
                               height=self.canvas_size, 
                               bg='white', 
                               cursor='pencil')
        self.canvas.pack(padx=5, pady=5)
        
        # 绑定鼠标事件
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)
        
        # 控制按钮框架
        button_frame = tk.Frame(left_frame, bg='#2c3e50')
        button_frame.pack(pady=20)
          # 预测按钮
        self.predict_btn = tk.Button(button_frame, text="CNN识别", 
                                   command=self.predict_digit,
                                   font=self.button_font,
                                   bg='#27ae60', fg='white',
                                   width=12, height=2,
                                   relief='raised', bd=2)
        self.predict_btn.pack(side='left', padx=5)
        
        # 清除按钮
        self.clear_btn = tk.Button(button_frame, text="清除画布", 
                                 command=self.clear_canvas,
                                 font=self.button_font,
                                 bg='#e74c3c', fg='white',
                                 width=12, height=2,
                                 relief='raised', bd=2)
        self.clear_btn.pack(side='left', padx=5)
          # 笔刷控制框架
        brush_frame = tk.Frame(left_frame, bg='#2c3e50')
        brush_frame.pack(pady=15)
        
        brush_label = tk.Label(brush_frame, text="笔刷大小:", 
                              font=self.default_font, bg='#2c3e50', fg='#ecf0f1')
        brush_label.pack(side='left')
        
        self.brush_var = tk.IntVar(value=self.brush_size)
        brush_scale = tk.Scale(brush_frame, from_=10, to=40, 
                             orient='horizontal', variable=self.brush_var,
                             command=self.update_brush_size,
                             bg='#34495e', fg='#ecf0f1', 
                             highlightbackground='#2c3e50')
        brush_scale.pack(side='left', padx=10)
        
        # 右侧框架（结果显示）
        right_frame = tk.Frame(main_frame, bg='#2c3e50')
        right_frame.pack(side='right', fill='both', padx=(20, 0))
        
        # 预测结果区域
        result_frame = tk.Frame(right_frame, bg='#34495e', relief='raised', bd=3)
        result_frame.pack(fill='x', pady=(0, 20))
        
        result_title = tk.Label(result_frame, text="🎯 预测结果", 
                               font=('Arial', 16, 'bold'), 
                               bg='#34495e', fg='#ecf0f1')
        result_title.pack(pady=10)
        
        # 预测数字显示
        self.result_label = tk.Label(result_frame, text="?", 
                                   font=('Arial', 56, 'bold'), 
                                   bg='#ecf0f1', fg='#e74c3c',
                                   width=3, height=1, relief='sunken', bd=3)
        self.result_label.pack(pady=15)
        
        # 置信度标签
        self.confidence_label = tk.Label(result_frame, text="置信度: --", 
                                       font=('Arial', 12, 'bold'), 
                                       bg='#34495e', fg='#f39c12')
        self.confidence_label.pack(pady=(0, 15))
        
        # 概率分布图框架
        plot_frame = tk.Frame(right_frame, bg='#34495e', relief='raised', bd=3)
        plot_frame.pack(fill='both', expand=True)
        
        plot_title = tk.Label(plot_frame, text="📊 概率分布", 
                             font=('Arial', 14, 'bold'), 
                             bg='#34495e', fg='#ecf0f1')
        plot_title.pack(pady=10)
        
        # 概率分布图
        self.create_probability_plot(plot_frame)
        
        # 说明文字
        instruction_frame = tk.Frame(right_frame, bg='#2c3e50')
        instruction_frame.pack(pady=20, fill='x')
        
        instruction_text = ("💡 使用说明:\\n"
                          "1. 在画布上画一个数字 (0-9)\\n"
                          "2. 点击 '🎯 CNN识别' 获得预测\\n"
                          "3. 查看置信度和概率分布\\n"
                          "4. CNN模型具有更高的准确率")
        
        instruction_label = tk.Label(instruction_frame, text=instruction_text, 
                                   font=('Arial', 9), 
                                   bg='#2c3e50', fg='#bdc3c7',
                                   justify='left')
        instruction_label.pack()
        
        # 性能信息
        perf_text = f"⚡ 设备: {self.device.type.upper()}"
        if torch.cuda.is_available():
            perf_text += f" ({torch.cuda.get_device_name()})"
        
        perf_label = tk.Label(instruction_frame, text=perf_text, 
                             font=('Arial', 8), 
                             bg='#2c3e50', fg='#95a5a6')
        perf_label.pack(pady=(10, 0))
        
    def create_probability_plot(self, parent):
        """创建概率分布图"""
        # matplotlib图形 - 深色主题
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(4.5, 3))
        self.fig.patch.set_facecolor('#34495e')
        self.ax.set_facecolor('#2c3e50')
        
        self.ax.set_xlabel('数字', color='#ecf0f1', fontsize=10)
        self.ax.set_ylabel('概率', color='#ecf0f1', fontsize=10)
        self.ax.set_title('CNN预测概率', color='#ecf0f1', fontsize=12, fontweight='bold')
        self.ax.set_ylim(0, 1)
        self.ax.tick_params(colors='#ecf0f1', labelsize=8)
        
        # 初始化空的柱状图
        self.bars = self.ax.bar(range(10), [0]*10, 
                               color='#3498db', edgecolor='#ecf0f1', alpha=0.7)
        self.ax.set_xticks(range(10))
        self.ax.grid(True, alpha=0.3, color='#7f8c8d')
        
        # 嵌入到tkinter
        self.canvas_plot = FigureCanvasTkAgg(self.fig, parent)
        self.canvas_plot.get_tk_widget().pack(pady=10, padx=10, fill='both', expand=True)
        
    def update_brush_size(self, value):
        """更新笔刷大小"""
        self.brush_size = int(value)
        
    def start_drawing(self, event):
        """开始绘画"""
        self.last_x = event.x
        self.last_y = event.y
        
    def draw(self, event):
        """绘画过程"""
        if hasattr(self, 'last_x') and hasattr(self, 'last_y'):
            # 在tkinter画布上绘制
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                  width=self.brush_size, fill='black', capstyle='round')
            
            # 在PIL图像上绘制
            self.canvas_draw.line([self.last_x, self.last_y, event.x, event.y],
                                fill='black', width=self.brush_size)
            
        self.last_x = event.x
        self.last_y = event.y
        
    def stop_drawing(self, event):
        """停止绘画"""
        if hasattr(self, 'last_x'):
            del self.last_x
        if hasattr(self, 'last_y'):
            del self.last_y
            
    def clear_canvas(self):
        """清除画布"""
        self.canvas.delete("all")
        self.canvas_image = Image.new("RGB", (self.canvas_size, self.canvas_size), "white")
        self.canvas_draw = ImageDraw.Draw(self.canvas_image)
        
        # 重置结果显示
        self.result_label.config(text="?")
        self.confidence_label.config(text="置信度: --")
        
        # 清除概率图
        for bar in self.bars:
            bar.set_height(0)
            bar.set_color('#3498db')
        self.canvas_plot.draw()
        
    def preprocess_image(self):
        """预处理图像为CNN模型输入格式"""
        # 转换为灰度图
        img = self.canvas_image.convert('L')
        
        # 反转颜色（MNIST是黑底白字）
        img = Image.eval(img, lambda x: 255 - x)
        
        # 调整大小到28x28
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # 转换为numpy数组并标准化
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # 标准化（使用MNIST的均值和标准差）
        img_array = (img_array - 0.1307) / 0.3081
        
        # 添加batch维度和通道维度 [1, 1, 28, 28]
        img_tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)
        
        return img_tensor
        
    def predict_digit(self):
        """使用CNN预测数字"""
        if self.model is None:
            messagebox.showerror("错误", "CNN模型未加载，请先训练模型")
            return
            
        try:
            # 预处理图像
            img_tensor = self.preprocess_image().to(self.device)
            
            # CNN预测
            with torch.no_grad():
                output = self.model(img_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted = torch.argmax(output, dim=1)
                confidence = torch.max(probabilities, dim=1)[0]
                
            # 显示结果
            predicted_digit = predicted.item()
            confidence_score = confidence.item()
            
            self.result_label.config(text=str(predicted_digit))
            self.confidence_label.config(text=f"置信度: {confidence_score:.1%}")
            
            # 更新概率分布图
            probs = probabilities.cpu().numpy()[0]
            for i, bar in enumerate(self.bars):
                bar.set_height(probs[i])
                # 高亮预测的数字
                if i == predicted_digit:
                    bar.set_color('#e74c3c')  # 红色高亮
                else:
                    bar.set_color('#3498db')  # 蓝色
                    
            # 更新图表标题显示置信度
            self.ax.set_title(f'CNN预测: {predicted_digit} (置信度: {confidence_score:.1%})', 
                            color='#ecf0f1', fontsize=12, fontweight='bold')
            self.canvas_plot.draw()
            
        except Exception as e:
            messagebox.showerror("错误", f"CNN预测失败: {e}")

def main():
    """主函数"""
    root = tk.Tk()
    app = CNNDigitRecognitionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

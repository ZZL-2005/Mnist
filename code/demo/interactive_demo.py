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

# 定义网络结构（必须与训练时一致）
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

class DigitRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("手写数字识别演示")
        self.root.geometry("900x600")
        self.root.configure(bg='#f0f0f0')
          # 加载模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()
        
        # 设置中文字体
        try:
            self.default_font = font.Font(family="Microsoft YaHei", size=10)
            self.title_font = font.Font(family="Microsoft YaHei", size=20, weight="bold")
            self.button_font = font.Font(family="Microsoft YaHei", size=12, weight="bold")
            self.result_font = font.Font(family="Microsoft YaHei", size=48, weight="bold")
            self.label_font = font.Font(family="Microsoft YaHei", size=12)
        except:
            # 如果微软雅黑不可用，使用默认字体
            self.default_font = font.Font(size=10)
            self.title_font = font.Font(size=20, weight="bold")
            self.button_font = font.Font(size=12, weight="bold")
            self.result_font = font.Font(size=48, weight="bold")
            self.label_font = font.Font(size=12)
        
        # 画布设置
        self.canvas_size = 280
        self.brush_size = 20
        
        # 创建界面
        self.create_widgets()
        
        # 画布图像
        self.canvas_image = Image.new("RGB", (self.canvas_size, self.canvas_size), "white")
        self.canvas_draw = ImageDraw.Draw(self.canvas_image)
        
    def load_model(self):
        """加载训练好的模型"""
        try:
            model = DigitClassifier()
            model.load_state_dict(torch.load('./models/digit_classifier.pth', map_location=self.device))
            model.to(self.device)
            model.eval()
            print("模型加载成功!")
            return model
        except Exception as e:
            messagebox.showerror("错误", f"无法加载模型: {e}\\n请先运行 train_digit_classifier.py 训练模型")
            return None
    
    def create_widgets(self):
        """创建GUI组件"""
        # 主框架
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        # 左侧框架（画布和控制）
        left_frame = tk.Frame(main_frame, bg='#f0f0f0')
        left_frame.pack(side='left', fill='both', expand=True)
          # 标题
        title_label = tk.Label(left_frame, text="手写数字识别", 
                              font=self.title_font, 
                              bg='#f0f0f0', fg='#333333')
        title_label.pack(pady=(0, 20))
        
        # 画布框架
        canvas_frame = tk.Frame(left_frame, relief='sunken', bd=2)
        canvas_frame.pack(pady=10)
        
        # 绘图画布
        self.canvas = tk.Canvas(canvas_frame, 
                               width=self.canvas_size, 
                               height=self.canvas_size, 
                               bg='white', 
                               cursor='pencil')
        self.canvas.pack()
        
        # 绑定鼠标事件
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)
        
        # 控制按钮框架
        button_frame = tk.Frame(left_frame, bg='#f0f0f0')
        button_frame.pack(pady=20)
          # 预测按钮
        self.predict_btn = tk.Button(button_frame, text="识别数字", 
                                   command=self.predict_digit,
                                   font=self.button_font,
                                   bg='#4CAF50', fg='white',
                                   width=12, height=2)
        self.predict_btn.pack(side='left', padx=5)
        
        # 清除按钮
        self.clear_btn = tk.Button(button_frame, text="清除画布", 
                                 command=self.clear_canvas,
                                 font=self.button_font,
                                 bg='#f44336', fg='white',
                                 width=12, height=2)
        self.clear_btn.pack(side='left', padx=5)
          # 笔刷大小调节
        brush_frame = tk.Frame(left_frame, bg='#f0f0f0')
        brush_frame.pack(pady=10)
        
        tk.Label(brush_frame, text="笔刷大小:", 
                font=self.default_font, bg='#f0f0f0').pack(side='left')
        
        self.brush_var = tk.IntVar(value=self.brush_size)
        brush_scale = tk.Scale(brush_frame, from_=10, to=40, 
                             orient='horizontal', variable=self.brush_var,
                             command=self.update_brush_size)
        brush_scale.pack(side='left', padx=5)
        
        # 右侧框架（结果显示）
        right_frame = tk.Frame(main_frame, bg='#f0f0f0')
        right_frame.pack(side='right', fill='both', padx=(20, 0))
          # 预测结果标题
        result_title = tk.Label(right_frame, text="预测结果", 
                               font=self.label_font, 
                               bg='#f0f0f0', fg='#333333')
        result_title.pack(pady=(0, 10))
        
        # 预测数字显示
        self.result_label = tk.Label(right_frame, text="?", 
                                   font=self.result_font, 
                                   bg='white', fg='#2196F3',
                                   width=3, height=2, relief='sunken', bd=2)
        self.result_label.pack(pady=10)
          # 置信度标签
        self.confidence_label = tk.Label(right_frame, text="置信度: --", 
                                       font=self.label_font, 
                                       bg='#f0f0f0', fg='#666666')
        self.confidence_label.pack(pady=5)
        
        # 概率分布图
        self.create_probability_plot(right_frame)
          # 说明文字
        instruction_text = ("使用说明:\\n"
                          "1. 在左侧白色画布上画一个数字\\n"
                          "2. 点击'识别数字'按钮进行预测\\n"
                          "3. 可以调节笔刷大小\\n"
                          "4. 右侧显示预测结果和概率分布")
        
        instruction_label = tk.Label(right_frame, text=instruction_text, 
                                   font=self.default_font, 
                                   bg='#f0f0f0', fg='#666666',
                                   justify='left')
        instruction_label.pack(pady=20, anchor='w')
        
    def create_probability_plot(self, parent):
        """创建概率分布图"""
        # matplotlib图形
        self.fig, self.ax = plt.subplots(figsize=(4, 3))
        self.ax.set_xlabel('数字')
        self.ax.set_ylabel('概率')
        self.ax.set_title('预测概率分布')
        self.ax.set_ylim(0, 1)
        
        # 初始化空的柱状图
        self.bars = self.ax.bar(range(10), [0]*10, color='lightblue', edgecolor='navy')
        self.ax.set_xticks(range(10))
        
        # 嵌入到tkinter
        self.canvas_plot = FigureCanvasTkAgg(self.fig, parent)
        self.canvas_plot.get_tk_widget().pack(pady=10)
        
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
        self.canvas_plot.draw()
        
    def preprocess_image(self):
        """预处理图像为模型输入格式"""
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
        
        # 添加batch维度和通道维度
        img_tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)
        
        return img_tensor
        
    def predict_digit(self):
        """预测数字"""
        if self.model is None:
            messagebox.showerror("错误", "模型未加载，请先训练模型")
            return
            
        try:
            # 预处理图像
            img_tensor = self.preprocess_image().to(self.device)
            
            # 预测
            with torch.no_grad():
                output = self.model(img_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted = torch.argmax(output, dim=1)
                confidence = torch.max(probabilities, dim=1)[0]
                
            # 显示结果
            predicted_digit = predicted.item()
            confidence_score = confidence.item()
            
            self.result_label.config(text=str(predicted_digit))
            self.confidence_label.config(text=f"置信度: {confidence_score:.2%}")
            
            # 更新概率分布图
            probs = probabilities.cpu().numpy()[0]
            for i, bar in enumerate(self.bars):
                bar.set_height(probs[i])
                # 高亮预测的数字
                if i == predicted_digit:
                    bar.set_color('red')
                else:
                    bar.set_color('lightblue')
                    
            self.canvas_plot.draw()
            
        except Exception as e:
            messagebox.showerror("错误", f"预测失败: {e}")

def main():
    """主函数"""
    root = tk.Tk()
    app = DigitRecognitionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

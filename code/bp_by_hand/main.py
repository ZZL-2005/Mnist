# manual_mlp.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json
import time
from datetime import datetime
# convert_to_npz.py
import torch
import numpy as np

# ---------- 加载数据 ----------
data = np.load(r"subset_data\mnist_fixed_subset_npz.npz")
train_images = data['train_images'].reshape(-1, 784)
train_labels = data['train_labels']
test_images = data['test_images'].reshape(-1, 784)
test_labels = data['test_labels']

# ---------- 超参数 ----------
input_dim = 784
hidden1 = 30
output_dim = 10
lr = 0.01
epochs = 80
batch_size = 256

# ---------- 初始化权重 ----------
W1 = np.random.randn(input_dim, hidden1) * np.sqrt(2. / input_dim)
b1 = np.zeros((1, hidden1))
W2 = np.random.randn(hidden1, output_dim) * np.sqrt(2. / hidden1)
b2 = np.zeros((1, output_dim))

# ---------- 激活函数 ----------
def relu(x): return np.maximum(0, x)
def relu_deriv(x): return (x > 0).astype(float)
def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)  # 防止溢出
    exps = np.exp(x)
    return exps / np.sum(exps, axis=1, keepdims=True)
def cross_entropy(pred, labels):
    m = labels.shape[0]
    return -np.mean(np.log(pred[np.arange(m), labels] + 1e-9))
def accuracy(pred, labels):
    return np.mean(np.argmax(pred, axis=1) == labels)

# ---------- 训练 ----------
train_acc_list, test_acc_list = [], []
train_loss_list, test_loss_list = [], []
training_log = {
    'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),    'hyperparameters': {
        'input_dim': input_dim,
        'hidden1': hidden1,
        'output_dim': output_dim,
        'lr': lr,
        'epochs': epochs,
        'batch_size': batch_size
    },
    'training_history': []
}

start_time = time.time()

for epoch in range(epochs):
    epoch_start = time.time()
    epoch_losses = []
    
    perm = np.random.permutation(len(train_images))
    for i in range(0, len(train_images), batch_size):
        idx = perm[i:i+batch_size]
        X = train_images[idx]
        Y = train_labels[idx]        # forward
        z1 = X @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 + b2
        out = softmax(z2)
        
        # 计算损失
        batch_loss = cross_entropy(out, Y)
        epoch_losses.append(batch_loss)

        # backward
        m = Y.shape[0]
        delta2 = out
        delta2[np.arange(m), Y] -= 1
        delta2 /= m

        dW2 = a1.T @ delta2
        db2 = np.sum(delta2, axis=0, keepdims=True)

        delta1 = (delta2 @ W2.T) * relu_deriv(z1)
        dW1 = X.T @ delta1
        db1 = np.sum(delta1, axis=0, keepdims=True)

        # 更新参数
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2    # 评估准确率和损失
    def eval_model(images, labels):
        h1 = relu(images @ W1 + b1)
        out = softmax(h1 @ W2 + b2)
        acc = accuracy(out, labels)
        loss = cross_entropy(out, labels)
        return acc, loss, out

    train_acc, train_loss, train_pred = eval_model(train_images, train_labels)
    test_acc, test_loss, test_pred = eval_model(test_images, test_labels)
    
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)
    
    epoch_time = time.time() - epoch_start
    
    # 记录训练日志
    epoch_log = {
        'epoch': epoch + 1,
        'train_acc': float(train_acc),
        'test_acc': float(test_acc),
        'train_loss': float(train_loss),
        'test_loss': float(test_loss),
        'epoch_time': float(epoch_time),
        'avg_batch_loss': float(np.mean(epoch_losses))
    }
    training_log['training_history'].append(epoch_log)
    
    print(f"Epoch {epoch+1:2d} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | "
          f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Time: {epoch_time:.2f}s")

total_time = time.time() - start_time
training_log['total_time'] = float(total_time)
training_log['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# ---------- 保存训练日志 ----------
log_filename = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(log_filename, 'w') as f:
    json.dump(training_log, f, indent=2)
print(f"\n训练日志已保存到: {log_filename}")

# ---------- 绘制训练曲线 ----------
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 左图: 损失曲线
axes[0].plot(range(1, epochs+1), train_loss_list, label='Train Loss', marker='o', color='red')
axes[0].plot(range(1, epochs+1), test_loss_list, label='Test Loss', marker='s', color='orange')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Test Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 右图: 准确率曲线
axes[1].plot(range(1, epochs+1), train_acc_list, label='Train Acc', marker='o', color='blue')
axes[1].plot(range(1, epochs+1), test_acc_list, label='Test Acc', marker='s', color='green')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training and Test Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"training_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", dpi=300, bbox_inches='tight')
plt.show()

# ---------- 计算最终预测结果 ----------
def get_predictions(images):
    h1 = relu(images @ W1 + b1)
    out = softmax(h1 @ W2 + b2)
    return np.argmax(out, axis=1)

train_predictions = get_predictions(train_images)
test_predictions = get_predictions(test_images)

# ---------- 绘制混淆矩阵 ----------
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 训练集混淆矩阵
train_cm = confusion_matrix(train_labels, train_predictions)
sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title(f'Training Set Confusion Matrix\nAccuracy: {train_acc_list[-1]:.4f}')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')

# 测试集混淆矩阵
test_cm = confusion_matrix(test_labels, test_predictions)
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title(f'Test Set Confusion Matrix\nAccuracy: {test_acc_list[-1]:.4f}')
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('True Label')

plt.tight_layout()
plt.savefig(f"confusion_matrices_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", dpi=300, bbox_inches='tight')
plt.show()

# ---------- 打印总结信息 ----------
print(f"\n{'='*50}")
print(f"训练完成! 总用时: {total_time:.2f}秒")
print(f"最终训练准确率: {train_acc_list[-1]:.4f}")
print(f"最终测试准确率: {test_acc_list[-1]:.4f}")
print(f"最终训练损失: {train_loss_list[-1]:.4f}")
print(f"最终测试损失: {test_loss_list[-1]:.4f}")
print(f"{'='*50}")

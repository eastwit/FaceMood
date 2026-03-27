import json
import matplotlib.pyplot as plt
import numpy as np

# 读取训练指标数据
with open('./data/training_metrics.json', 'r') as f:
    data = json.load(f)

history = data['history']
epochs = history['epochs']
train_acc = history['train_acc']
val_acc = history['val_acc']

# 从 step_loss 中提取每个 epoch 的平均损失
epoch_losses = {}
for item in history['step_loss']:
    ep = item['epoch']
    if ep not in epoch_losses:
        epoch_losses[ep] = []
    epoch_losses[ep].append(item['loss'])

# 计算每个 epoch 的平均损失
avg_losses = [np.mean(epoch_losses[e]) for e in epochs]

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 创建图表
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 图1: 准确率对比
axes[0].plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=1.5)
axes[0].plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=1.5)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy (%)', fontsize=12)
axes[0].set_title('Training vs Validation Accuracy', fontsize=14)
axes[0].legend(loc='lower right', fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(1, len(epochs))

# 图2: 训练损失曲线
axes[1].plot(epochs, avg_losses, 'g-', linewidth=1.5)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].set_title('Training Loss Curve', fontsize=14)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(1, len(epochs))

# 图3: 验证准确率（放大版）
axes[2].plot(epochs, val_acc, 'r-', linewidth=1.5, marker='o', markersize=2)
axes[2].set_xlabel('Epoch', fontsize=12)
axes[2].set_ylabel('Validation Accuracy (%)', fontsize=12)
axes[2].set_title('Validation Accuracy Detail', fontsize=14)
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim(1, len(epochs))

# 标注最高点
best_epoch = epochs[np.argmax(val_acc)]
best_val_acc = max(val_acc)
axes[2].annotate(f'Best: {best_val_acc:.2f}%\n(Epoch {best_epoch})',
                 xy=(best_epoch, best_val_acc),
                 xytext=(best_epoch + 10, best_val_acc - 2),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=10, color='red')

plt.tight_layout()
plt.savefig('./data/training_curves.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n训练完成！")
print(f"最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch})")
print(f"图表已保存至: ./data/training_curves.png")

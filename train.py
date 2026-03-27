import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

# 导入你的定义
from dataset import FER2013ForXception
from facecnn import MiniXception


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    BATCH_SIZE = 64
    EPOCHS = 100  
    LEARNING_RATE = 0.0005  
    DATA_PATH = './data'

    EXISTING_MODEL = r'D:\UESTC\UESTC_6\人机\project1\models\mini_xception_fer2013.pth'
    SAVE_PATH = './models/best_model_continued.pth'
    CHECKPOINT_PATH = './models/checkpoint_continued.pth'

    if not os.path.exists('./models'):
        os.makedirs('./models')

    from torchvision import transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),  
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = FER2013ForXception(root=DATA_PATH, mode='train', transform=train_transform)
    val_dataset = FER2013ForXception(root=DATA_PATH, mode='val', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = MiniXception(input_shape=(1, 48, 48), num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    start_epoch = 0
    best_acc = 0.0

    # 优先检查是否存在本次运行生成的断点
    if os.path.exists(CHECKPOINT_PATH):
        print(f"--- 发现本次训练断点，正在恢复进度... ---")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt['best_acc']
        print(f"已恢复至第 {start_epoch + 1} 轮。")
    elif os.path.exists(EXISTING_MODEL):
        print(f"--- 正在加载外部权重文件进行微调: {EXISTING_MODEL} ---")
        weights = torch.load(EXISTING_MODEL, map_location=device)
        # 兼容性处理：防止加载的是带字典的断点文件
        if isinstance(weights, dict) and 'model_state_dict' in weights:
            model.load_state_dict(weights['model_state_dict'])
            best_acc = weights.get('best_acc', 0.0)
        else:
            model.load_state_dict(weights)
        print("外部权重加载成功，开始继续训练！")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{EPOCHS}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        train_acc = 100. * correct / total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, lbs in val_loader:
                imgs, lbs = imgs.to(device), lbs.to(device)
                out = model(imgs)
                _, pre = out.max(1)
                val_total += lbs.size(0)
                val_correct += pre.eq(lbs).sum().item()

        val_acc = 100. * val_correct / val_total
        print(f'==> Epoch {epoch + 1}: 训练 Acc: {train_acc:.2f}%, 验证 Acc: {val_acc:.2f}%')
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': max(val_acc, best_acc)
        }
        torch.save(checkpoint_data, CHECKPOINT_PATH)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f'⭐️ 达到新的最高准确率，已保存至 {SAVE_PATH}')

        scheduler.step()

    print(f'继续训练任务结束！')


if __name__ == '__main__':
    train()
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json

# 导入你的定义
from dataset import FER2013ForXception
from facecnn import MiniXception


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total_samples if total_samples else 0.0
    acc = 100.0 * total_correct / total_samples if total_samples else 0.0
    return avg_loss, acc


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    BATCH_SIZE = 64
    EPOCHS = 100  
    LEARNING_RATE = 0.0005  
    DATA_PATH = './data'
    START_FROM_SCRATCH = True

    SAVE_PATH = './models/best_model_continued.pth'
    CHECKPOINT_PATH = './models/checkpoint_continued.pth'
    METRICS_JSON_PATH = './models/training_metrics.json'

    os.makedirs('./models', exist_ok=True)

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
    best_epoch = -1

    metrics = {
        'start_from_scratch': START_FROM_SCRATCH,
        'epochs': [],
        'train_acc': [],
        'val_acc': [],
        'step_loss': []
    }

    if not START_FROM_SCRATCH and os.path.exists(CHECKPOINT_PATH):
        print(f"--- 发现本次训练断点，正在恢复进度... ---")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt['best_acc']
        best_epoch = ckpt.get('best_epoch', -1)
        print(f"已恢复至第 {start_epoch + 1} 轮。")
    elif START_FROM_SCRATCH:
        print("--- 从头开始训练（忽略已有权重和断点）---")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        correct, total = 0, 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics['step_loss'].append({
                'epoch': epoch + 1,
                'step': i + 1,
                'total_steps': len(train_loader),
                'loss': float(loss.item())
            })

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{EPOCHS}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        train_acc = 100.0 * correct / total if total else 0.0

        _, val_acc = evaluate(model, val_loader, criterion, device)

        metrics['epochs'].append(epoch + 1)
        metrics['train_acc'].append(train_acc)
        metrics['val_acc'].append(val_acc)

        print(
            f'==> Epoch {epoch + 1}: '
            f'train_acc={train_acc:.2f}%, val_acc={val_acc:.2f}%'
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), SAVE_PATH)
            print(f'⭐️ 达到新的最高准确率，已保存至 {SAVE_PATH}')

        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
            'best_epoch': best_epoch
        }
        torch.save(checkpoint_data, CHECKPOINT_PATH)

        scheduler.step()

    output_data = {
        'config': {
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'data_path': DATA_PATH,
            'start_from_scratch': START_FROM_SCRATCH
        },
        'history': metrics,
        'best_val_acc': best_acc,
        'best_epoch': best_epoch,
        'final_train_acc': metrics['train_acc'][-1] if metrics['train_acc'] else None,
        'final_val_acc': metrics['val_acc'][-1] if metrics['val_acc'] else None
    }

    with open(METRICS_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f'训练结束，指标已保存至: {METRICS_JSON_PATH}')


if __name__ == '__main__':
    train()

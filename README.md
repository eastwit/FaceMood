# FaceMood - 实时表情识别与语音反馈系统

一个基于深度学习的实时人脸表情识别系统，能够通过摄像头检测用户表情并给予语音反馈。

## 快速开始

按下面流程可以快速运行主程序 `facemood.py`：

1. 安装依赖

```bash
git clone https://github.com/eastwit/FaceMood.git

cd FaceMood

pip install -r requirements.txt
```

2. 准备数据与权重

- 将 `fer2013.csv` 放到 `data/` 目录
- 将 `yolov8n-face-lindevs.pt` 和 `best_model_continued.pth` 放到 `weight/` 目录

> 或者直接`git lfs pull`

3. 运行实时表情识别（带语音反馈）

```bash
python facemood.py
```

运行后会打开摄像头，识别到稳定表情后自动语音播报；按 `Q` 键退出。

## 功能特性

- **实时人脸检测**：使用 YOLOv8 模型进行快速准确的人脸检测
- **表情识别**：支持 7 种基本表情分类
  - Angry（愤怒）
  - Disgust（厌恶）
  - Fear（恐惧）
  - Happy（开心）
  - Sad（悲伤）
  - Surprise（惊讶）
  - Neutral（中性）
- **语音反馈**：检测到稳定表情后自动播报语音反馈
- **断点续训**：训练过程支持断点保存与恢复

## 项目结构

```
FaceMood/
├── facemood.py      # 主程序 - 实时表情识别与语音反馈
├── test.py          # 测试脚本 - 基础表情识别演示
├── train.py         # 训练脚本 - MiniXception 模型训练
├── facecnn.py       # 模型定义 - MiniXception 神经网络
├── dataset.py       # 数据集加载 - FER2013 数据集处理
├── yolov8.py        # 人脸检测器 - YOLOv8 封装
├── sound.py         # 语音合成 - Edge TTS 语音播报
├── data/            # 数据目录
│   └── fer2013.csv  # FER2013 表情数据集
├── weight/          # 模型权重目录
│   ├── yolov8n-face-lindevs.pt    # YOLO 人脸检测模型
│   └── best_model_continued.pth  # 训练好的表情识别模型
├── models/          # 训练输出目录
└── README.md        # 项目说明文档
```

## 环境要求

- Python 3.8+
- PyTorch 1.10+
- OpenCV (cv2)
- Ultralytics (YOLOv8)
- Pandas
- edge-tts
- pygame

## 安装依赖

```bash
pip install torch torchvision opencv-python ultralytics pandas edge-tts pygame
```

## 数据集

项目使用 [FER2013](https://www.kaggle.com/datasets/msambare/fer2013) 数据集，包含 48x48 像素的灰度人脸图像，涵盖 7 种表情类别。

将 `fer2013.csv` 文件放置在 `data/` 目录下。

## 模型权重

### YOLOv8 人脸检测模型

下载 YOLOv8 人脸检测权重并放置在 `weight/` 目录：

```
weight/yolov8n-face-lindevs.pt
```

### 表情识别模型

训练后，模型权重保存在：

```
weight/best_model_continued.pth
```

## 使用方法

### 1. 训练模型

```bash
python train.py
```

训练脚本支持：
- 从外部预训练模型继续训练
- 断点恢复训练
- 学习率调度
- 数据增强（随机翻转、旋转）

### 2. 实时表情识别（带语音反馈）

```bash
python facemood.py
```

功能：
- 打开摄像头实时检测
- 显示人脸边界框和识别结果
- 当表情稳定显示 15 帧后，自动播报语音反馈
- 按 `Q` 键退出

### 3. 基础表情识别测试

```bash
python test.py
```

仅显示识别结果，不包含语音功能。

### 4. 语音播报测试

```bash
python sound.py
```

测试 Edge TTS 语音合成功能。

### 5. 独立人脸检测

```bash
python yolov8.py
```

仅使用 YOLOv8 进行人脸检测，不进行表情识别。

## 模型架构

### MiniXception

轻量级的 Xception 网络变体，专为 48x48 灰度图像设计：

- 入口卷积层
- 4 个 Xception 模块（包含深度可分离卷积）
- 全局平均池化
- 7 类输出

### YOLOv8

用于快速人脸检测，提供 bounding box 定位。

## 语音反馈

系统使用微软 Edge TTS 进行语音合成，支持中文语音（zh-CN-YunxiNeural）。

表情对应的反馈文案：

| 表情 | 语音反馈 |
|------|----------|
| Angry | 你看起来有点生气，深呼吸放松一下 |
| Disgust | 你看起来有些厌恶 |
| Fear | 你看起来有点害怕 |
| Happy | 你看起来很开心 |
| Sad | 你看起来有些不开心，发生什么事了？ |
| Surprise | 什么事情让你这么惊讶？ |
| Neutral | 你看起来很平静 |

## 技术栈

- **深度学习框架**: PyTorch
- **目标检测**: YOLOv8 (Ultralytics)
- **计算机视觉**: OpenCV
- **语音合成**: Edge TTS
- **数据集**: FER2013

## 注意事项

1. 首次运行需要下载 YOLO 模型权重
2. 确保摄像头权限已开启
3. 语音功能需要网络连接
4. GPU 可加速推理过程（自动检测 CUDA）

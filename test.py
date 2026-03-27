import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from ultralytics import YOLO
from facecnn import (MiniXception)


def start_camera():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = './weight/best_model_continued.pth'

    # 表情映射
    emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

    # YOLOv8 人脸检测器（权重放在 weight 目录）
    yolo_face = YOLO("./weight/yolov8n-face-lindevs.pt")

    # 加载模型
    model = MiniXception(input_shape=(1, 48, 48), num_classes=7).to(device)
    if torch.os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print("模型权重加载成功！")
    else:
        print("未找到模型文件，请先运行 train.py")
        return

    # 预处理变换
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    cap = cv2.VideoCapture(0)

    print("正在开启摄像头... 按 'Q' 退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 转为灰度图用于人脸检测
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # YOLOv8 检测人脸（cls==0 认为是 face 类）
        results = yolo_face(frame, conf=0.5, verbose=False)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                if int(box.cls[0]) != 0:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1] - 1, x2)
                y2 = min(frame.shape[0] - 1, y2)
                bw, bh = (x2 - x1), (y2 - y1)
                if bw <= 0 or bh <= 0:
                    continue

                # 1. 绘制人脸框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 2. 裁剪人脸区域 (ROI)
                roi_gray = gray_frame[y1:y2, x1:x2]
                if roi_gray.size == 0:
                    continue

                # 3. 预处理并送入模型
                try:
                    roi_pil = Image.fromarray(roi_gray)
                    roi_tensor = transform(roi_pil).unsqueeze(0).to(device)  # [1, 1, 48, 48]

                    with torch.no_grad():
                        outputs = model(roi_tensor)
                        _, predicted = torch.max(outputs, 1)
                        label = emotion_dict[predicted.item()]

                    # 4. 在画面上打字
                    cv2.putText(
                        frame,
                        label,
                        (x1 + 10, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                except Exception as e:
                    print(f"处理出错: {e}")

        # 显示画面
        cv2.imshow('Emotion Detector - Press Q to Quit', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    start_camera()
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

from ultralytics import YOLO
from facecnn import MiniXception 
from sound import speak, is_playing

EMOTION_FEEDBACK = {
    "Angry":    "你看起来有点生气，深呼吸放松一下",
    "Disgust":  "你看起来有些厌恶",
    "Fear":     "你看起来有点害怕",
    "Happy":    "你看起来很开心",
    "Sad":      "你看起来有些不开心，发生什么事了？",
    "Surprise": "什么事情让你这么惊讶？",
    "Neutral":  "你看起来很平静",
}

def start_camera():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = './weight/best_model_continued.pth'
    YOLO_PATH = './weight/yolov8n-face-lindevs.pt'

    emotion_dict = {
        0: "Angry", 1: "Disgust", 2: "Fear",
        3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"
    }

    # 加载模型
    if not os.path.exists(YOLO_PATH):
        print(f"错误: 找不到 YOLO 模型 {YOLO_PATH}")
        return
    yolo_face = YOLO(YOLO_PATH)

    model = MiniXception(input_shape=(1, 48, 48), num_classes=7).to(device)
    if os.path.exists(MODEL_PATH):
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

    # 初始化参数
    cap = cv2.VideoCapture(0)
    STABLE_THRESHOLD = 15      
    stable_counter = 0         # 当前连续帧计数
    current_label = None       # 当前正在检测的标签
    last_spoken_label = None   # 上一次播报完的标签，防止同一个表情无限循环播报

    print("正在开启摄像头... 按 'Q' 退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 检测人脸
        results = yolo_face(frame, conf=0.5, verbose=False)
        detected_this_frame = False

        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) != 0: 
                    continue

                detected_this_frame = True
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # 边界保护
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)

                # 绘制人脸框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 裁剪与表情识别
                roi_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                try:
                    roi_pil = Image.fromarray(roi_gray)
                    roi_tensor = transform(roi_pil).unsqueeze(0).to(device)

                    with torch.no_grad():
                        outputs = model(roi_tensor)
                        _, predicted = torch.max(outputs, 1)
                        label = emotion_dict[predicted.item()]

                    # 实时 UI 更新
                    cv2.putText(frame, label, (x1, max(0, y1 - 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                    # 检查当前是否正在播放语音
                    if not is_playing():
                        # 如果没在说话，开始/继续计数
                        if label == current_label:
                            stable_counter += 1
                        else:
                            current_label = label
                            stable_counter = 1
                        
                        if stable_counter >= STABLE_THRESHOLD:
                            if label != last_spoken_label:
                                feedback = EMOTION_FEEDBACK.get(label, "检测到表情")
                                print(f"[触发语音] {label} -> {feedback}")
                                speak(feedback)
                                
                                last_spoken_label = label
                                stable_counter = 0
                            else:

                                pass
                    else:
                        stable_counter = 0
                        
                except Exception as e:
                    print(f"识别过程出错: {e}")

        if not detected_this_frame:
            stable_counter = 0
            current_label = None

        cv2.imshow('FaceMood - Press Q to Quit', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    start_camera()
import cv2
from ultralytics import YOLO
import time


class FaceDetector:
    def __init__(self, model_name="yolov8n.pt"):
        """初始化检测器"""
        self.model = YOLO(model_name)
        self.cap = cv2.VideoCapture(0)
        self.latest_face_crops = []

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # type: ignore
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

    def detect(self, frame, conf_threshold=0.5):
        """执行检测"""
        results = self.model(frame, conf=conf_threshold, verbose=False)
        return results

    def draw_boxes(self, frame, results):
        """绘制检测框"""
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if int(box.cls[0]) == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"Face {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
        return frame

    def extract_face_crops(self, frame, results, margin=0.15, min_size=20):
        """输出当前帧的人脸截取图像，返回[{bbox, conf, image}]"""
        crops = []
        h, w = frame.shape[:2]

        for result in results:
            boxes = result.boxes
            for box in boxes:
                if int(box.cls[0]) != 0:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])

                bw = x2 - x1
                bh = y2 - y1
                if bw <= 0 or bh <= 0:
                    continue

                pad_x = int(bw * margin)
                pad_y = int(bh * margin)

                cx1 = max(0, x1 - pad_x)
                cy1 = max(0, y1 - pad_y)
                cx2 = min(w, x2 + pad_x)
                cy2 = min(h, y2 + pad_y)

                if (cx2 - cx1) < min_size or (cy2 - cy1) < min_size:
                    continue

                face_img = frame[cy1:cy2, cx1:cx2].copy()
                crops.append(
                    {
                        "bbox": (cx1, cy1, cx2, cy2),
                        "conf": conf,
                        "image": face_img,
                    }
                )

        return crops

    def run(self):
        """主运行循环"""
        fps_counter = 0
        fps = 0
        start_time = time.time()

        print("🎬 按 q 退出")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # 检测
            results = self.detect(frame)

            # 输出当前帧的人脸截取图像，供其他脚本直接使用
            self.latest_face_crops = self.extract_face_crops(frame, results)

            # 绘制
            frame = self.draw_boxes(frame, results)

            # 计算FPS
            fps_counter += 1
            if time.time() - start_time >= 1:
                fps = fps_counter
                fps_counter = 0
                start_time = time.time()

            cv2.putText(
                frame,
                f"FPS: {fps}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                "press q to exit",
                (400, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )

            # 显示
            cv2.imshow("YOLOv8 Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()


# 运行
if __name__ == "__main__":
    detector = FaceDetector(model_name="./yolov8n-face-lindevs.pt")
    detector.run()

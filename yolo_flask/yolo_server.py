"""
YOLO 目标检测服务器
POST 视频帧到 http://服务器IP:5151/detect，返回检测后的图像
"""

import logging
import threading

import cv2
import numpy as np
from flask import Flask, request, Response, jsonify
from ultralytics import YOLO

# ── 配置 ──────────────────────────────────────
MODEL_PATH     = "yolov10l.pt"
DEVICE         = "cuda"
CONF           = 0.5
JPEG_QUALITY   = 80
MAX_UPLOAD_MB  = 10

# ── 初始化 ────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

model = YOLO(MODEL_PATH).to(DEVICE)
lock  = threading.Lock()          # 保证多线程推理安全
logger.info("模型加载完成: %s", MODEL_PATH)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


# ── 工具函数 ──────────────────────────────────

def detect(frame: np.ndarray) -> np.ndarray:
    """YOLO 推理并在图像上绘制检测框，返回处理后的图像。"""
    with lock:
        results = model.predict(frame, conf=CONF, verbose=False)

    out = frame.copy()
    for r in results:
        if r.boxes is None:
            continue
        for cls_id, xy, conf in zip(
            r.boxes.cls.cpu().numpy().astype(int),
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.conf.cpu().numpy(),
        ):
            x1, y1, x2, y2 = map(int, xy)
            label = f"{r.names[cls_id]} {conf:.2f}"
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return out


# ── 路由 ──────────────────────────────────────

@app.route("/detect", methods=["POST"])
def detect_endpoint():
    file = request.files.get("frame")
    if file is None:
        return jsonify(error="缺少 frame 字段"), 400

    frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify(error="无法解码图像"), 400

    ret, buf = cv2.imencode(".jpg", detect(frame), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ret:
        return jsonify(error="图像编码失败"), 500

    return Response(buf.tobytes(), mimetype="image/jpeg",
                    headers={"Access-Control-Allow-Origin": "*"})


@app.route("/health")
def health():
    return jsonify(status="ok", model=MODEL_PATH)


# ── 启动 ──────────────────────────────────────

if __name__ == "__main__":
    logger.info("服务启动 → http://0.0.0.0:5151")
    app.run(host="0.0.0.0", port=5151, threaded=True)
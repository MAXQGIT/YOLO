"""
YOLO 实时检测客户端 (Windows)  |  依赖: pip install requests opencv-python
"""

import threading
import cv2
import numpy as np
import requests

SERVER  = "http://192.168.216.60:5151/detect"  # 改为服务器 IP
CAP_ID  = 0
QUALITY = 70
latest  = None   # 最新结果帧（主线程只读）

def send_loop(cap):
    global latest
    sess = requests.Session()
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
        try:
            r = sess.post(SERVER, files={"frame": buf.tobytes()}, timeout=5)
            if r.status_code == 200:
                img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    latest = img
        except Exception as e:
            print(e)

cap = cv2.VideoCapture(CAP_ID, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
threading.Thread(target=send_loop, args=(cap,), daemon=True).start()

while True:
    if latest is not None:
        cv2.imshow("YOLO  |  Q退出", latest)
    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
        break

cap.release()
cv2.destroyAllWindows()
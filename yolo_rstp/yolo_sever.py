import subprocess, threading, cv2, numpy as np, time, os, json
from queue import Queue, Empty
from ultralytics import YOLO

# windows系统启动方式，使用VLC播放器播放视频结果
# https://github.com/bluenviron/mediamtx
# ffmpeg -f dshow -i video="Integrated Camera" ^
#   -video_size 1280x720 -framerate 30 ^
#   -pix_fmt yuv420p ^
#   -c:v libx264 -preset ultrafast -tune zerolatency ^
#   -crf 20 -maxrate 3M -bufsize 2M ^
#   -g 60 -keyint_min 60 ^
#   -f rtsp -rtsp_transport tcp ^
#   rtsp://192.168.216.60:8554/cam1

# ================= 配置区 =================
RTSP_SRC = "rtsp://192.168.216.60:8554/cam1"  # 唯一的源流，AI和显示共用同一帧
RTSP_OUT = "rtsp://192.168.216.60:8554/yolo_out"  # 输出流
AI_SIZE = 640  # AI推理正方形边长(letterbox目标尺寸)
FPS = 30
RTSP_TRANSPORT = "tcp"  # tcp比udp更抗丢包，避免花屏/马赛克
CONF_THRES = 0.3  # 置信度过滤阈值
# ==========================================

latest_boxes = []
box_lock = threading.Lock()
frame_queue_ai = Queue(maxsize=1)  # 只保留最新一帧给AI线程，避免积压导致延迟
latest_hd_frame = None
hd_lock = threading.Lock()
stop_flag = threading.Event()

model = YOLO("yolov10l.pt").to('cuda')


def letterbox(img, new_size=640, color=(114, 114, 114)):
    """等比例缩放 + 灰边填充，保持长宽比不失真，用于精确还原AI坐标到原图。
    这是单流结构下框能精准贴合目标的关键：AI看到的和高清帧是同一张图，
    只是做了无损的等比缩放+填充，反算回去坐标是完全精确的，
    不会像"HD流单独拉一路、AI流单独拉一路"那样因为两路解码不同步、
    分辨率比例不同(直接resize导致拉伸)而错位。"""
    h, w = img.shape[:2]
    scale = min(new_size / h, new_size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    top = (new_size - nh) // 2
    left = (new_size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, scale, left, top


def probe_stream(url, timeout=10):
    """用ffprobe探测源流真实宽高，不依赖cv2.VideoCapture去猜测/协商分辨率。"""
    cmd = [
        "ffprobe", "-v", "error", "-rtsp_transport", RTSP_TRANSPORT,
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json", url
    ]
    out = subprocess.check_output(cmd, timeout=timeout)
    info = json.loads(out)["streams"][0]
    w, h = int(info["width"]), int(info["height"])
    print(f"🔍 探测到源流真实分辨率: {w}x{h}")
    return w, h


def start_ffmpeg_reader(url, w, h):
    """系统ffmpeg子进程拉流解码裸帧，比cv2.VideoCapture更稳定干净。"""
    cmd = [
        "ffmpeg", "-rtsp_transport", RTSP_TRANSPORT, "-i", url,
        "-an", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-vsync", "0", "-"
    ]
    return subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        bufsize=w * h * 3 * 2
    )


def capture_thread(w, h):
    """唯一的拉流+解码线程：AI和渲染共享同一路帧，从源头上避免双流不同步的问题。"""
    global latest_hd_frame
    frame_size = w * h * 3
    proc = start_ffmpeg_reader(RTSP_SRC, w, h)

    while not stop_flag.is_set():
        raw = proc.stdout.read(frame_size)
        if len(raw) != frame_size:
            print("⚠️ 拉流进程异常/断流，重连中...")
            try:
                proc.kill()
                proc.wait()
            except Exception:
                pass
            time.sleep(1)
            proc = start_ffmpeg_reader(RTSP_SRC, w, h)
            continue

        frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3)).copy()

        with hd_lock:
            latest_hd_frame = frame

        if frame_queue_ai.full():
            try:
                frame_queue_ai.get_nowait()
            except Empty:
                pass
        frame_queue_ai.put(frame)

    try:
        proc.kill()
    except Exception:
        pass


def ai_detect_thread():
    """本地YOLO推理线程：letterbox后推理，再按letterbox参数精确反算回原图坐标。"""
    global latest_boxes

    while not stop_flag.is_set():
        try:
            frame = frame_queue_ai.get(timeout=1)
        except Empty:
            continue

        canvas, scale, pad_left, pad_top = letterbox(frame, AI_SIZE)

        try:
            results = model(canvas, imgsz=AI_SIZE, conf=CONF_THRES, verbose=False)[0]
        except Exception as e:
            print(f"⚠️ 推理出错: {e}")
            continue
        mapped = []
        if results is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                # 反算letterbox：先去掉padding，再除以scale，精确还原到原始高清帧坐标
                x1 = (x1 - pad_left) / scale
                y1 = (y1 - pad_top) / scale
                x2 = (x2 - pad_left) / scale
                y2 = (y2 - pad_top) / scale
                label= results.names[cls]

                mapped.append([x1, y1, x2, y2,label, conf])

        with box_lock:
            latest_boxes = mapped


def start_ffmpeg_writer(w, h):
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
        "-pix_fmt", "yuv420p",
        "-crf", "18", "-maxrate", "8M", "-bufsize", "4M",
        "-g", str(FPS * 2),
        "-f", "rtsp", "-rtsp_transport", RTSP_TRANSPORT,
        RTSP_OUT
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def hd_render_thread(ffmpeg_proc):
    while not stop_flag.is_set():
        with hd_lock:
            frame = None if latest_hd_frame is None else latest_hd_frame.copy()
        if frame is None:
            time.sleep(0.01)
            continue

        with box_lock:
            boxes = latest_boxes.copy()

        for b in boxes:
            x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            cls, conf = b[4], b[5]
            color = (0, 255, 0) if conf > 0.6 else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{cls} {conf:.2f}"
            cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        try:
            ffmpeg_proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            print("❌ FFmpeg输出管道断开")
            break

        time.sleep(1 / FPS)


if __name__ == "__main__":
    real_w, real_h = probe_stream(RTSP_SRC)

    ffmpeg_out_proc = start_ffmpeg_writer(real_w, real_h)
    t_cap = threading.Thread(target=capture_thread, args=(real_w, real_h), daemon=True)
    t_ai = threading.Thread(target=ai_detect_thread, daemon=True)
    t_render = threading.Thread(target=hd_render_thread, args=(ffmpeg_out_proc,), daemon=True)

    t_cap.start();
    t_ai.start();
    t_render.start()
    print(f"🟢 单流拉取 -> AI(letterbox {AI_SIZE}, 本地推理) + 高清渲染 {real_w}x{real_h}")
    print(f"🟢 输出地址: {RTSP_OUT} | Ctrl+C 停止")

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n🛑 停止中...")
    finally:
        stop_flag.set()
        time.sleep(0.3)
        ffmpeg_out_proc.stdin.close()
        ffmpeg_out_proc.wait()

from ultralytics import YOLO
from flask import *
import flask
'''
YOLO官网https://docs.ultralytics.com/
本地运行后，打开浏览器输入http://192.168.216.74:5151/test?content=摄像头id   
0是电脑自带摄像头
'''
app = Flask(__name__)  # 实例化flask


import cv2

model = YOLO('yolov8n.pt')  #模型可以到https://docs.ultralytics.com/models/   自行下载

# cv2.namedWindow('camera')
def get_frame(input_text):

    cap = cv2.VideoCapture(int(input_text)) #输入摄像机视频捕捉设备 id
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame)
        for result in results:
            boxes = result.boxes
            xyxy = boxes.xyxy.tolist()
            cls_list = boxes.cls.tolist()
            for cls, xy in zip(cls_list, xyxy):
                label = result.names[int(cls)]
                cv2.rectangle(frame, (int(xy[0]), int(xy[1]), int(xy[2]), int(xy[3])), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(xy[0]), int(xy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        net, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        if frame is None:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed', methods=['POST', 'GET'])  # 这个地址返回视频流响应
def video_feed():
    input_text = flask.request.args.get('content')
    return Response(get_frame(input_text), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5151)

'''
下面是本地测试的代码逻辑
'''
#     cv2.imshow('Yolo Real-Time Detection', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

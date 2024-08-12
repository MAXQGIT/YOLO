from ultralytics import YOLO

'''
YOLO官网https://docs.ultralytics.com/
yolo 算法再训练下面是改成windows系统训练了，如果使用linux或者ubuntu系统根据实际情况修改workers参数
'''
# model = YOLO('yolov8n.pt')
# results = model.train(data="data.yaml", epochs=100, imgsz=640,workers=0)
'''
本地测试代码
'''
# from ultralytics import YOLO
# model = YOLO('runs/detect/train/weights/best.pt').to('cpu')
# results = model.predict('datasets/fire-8/test/images/img_109_jpg.rf.629f764413847f6fb5e374209ddfc283.jpg')
# # print(results[0].names)
# results[0].show()


import flask
from flask import Flask,send_file

server = Flask(__name__)
model = YOLO('runs/detect/train/weights/best.pt')


@server.route('/test', methods=['POST', 'GET'])
def chat_l():
    input_text = flask.request.args.get('content')
    # response, history = model.chat(tokenizer, "你好", history=[])
    # print(response)
    results = model.predict(input_text)
    image_path ="output.jpg"
    results[0].save("output.jpg")
    return send_file(image_path,mimetype='image/jpeg')


if __name__ == '__main__':
    server.run(host='0.0.0.0', port=5151)


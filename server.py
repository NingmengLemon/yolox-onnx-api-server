#!/usr/bin/env python3

# yolox-onnx-api-server
# License: AGPL-3.0
# Github: https://github.com/nkxingxh/yolox-onnx-api-server

import argparse
import base64
import os
import random
import time

from collections import deque

# 开始计时
start_time = time.time()
print('https://github.com/nkxingxh/yolox-onnx-api-server')
print('Loading libraries, please wait...')

import cv2
import numpy as np
import onnxruntime

# from yolox.data.data_augment import preproc as preprocess
# from yolox.data.datasets import COCO_CLASSES
# from yolox.utils import mkdir, multiclass_nms, demo_postprocess
from utils import mkdir, multiclass_nms, demo_postprocess , vis

from flask import Flask, request, jsonify
app = Flask(__name__)

# 全局请求记录队列，用于保存请求时间戳
request_times = deque()
rate_limit = None  # 每秒允许的最大请求数


def console_log(text):
    print('[' + time.strftime("%H:%M:%S", time.localtime()) + '] ' + text)


def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def make_parser():
    parser = argparse.ArgumentParser("yolox-onnx-api-server")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        # default="yolox.onnx",
        help="指定ONNX模型文件。",
    )
    parser.add_argument(
        "-l",
        "--labels",
        type=str,
        required=True,
        # default="yolox.onnx",
        help="分类标签文件。",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        help="可视化图片输出目录。为空则不保存可视化结果",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.3,
        help="全局置信度阈值。",
    )
    parser.add_argument(
        "-i",
        "--input_shape",
        type=str,
        default="640,640",
        help="指定推理的输入形状。",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=9656,
        help="HTTP服务器监听端口。",
    )
    parser.add_argument(
        "-k",
        "--key",
        type=str,
        default=None,
        help="API密钥。",
    )
    parser.add_argument(
        "-r",
        "--rate_limit",
        type=int,
        default=None,
        help="每秒允许的最大请求数",
    )
    return parser


# 检查速率限制
def check_rate_limit():
    current_time = time.time()
    # 从队列中清理1秒以前的请求记录
    while request_times and current_time - request_times[0] > 1:
        request_times.popleft()

    # 如果记录的请求数超过限制，拒绝请求
    if len(request_times) >= rate_limit:
        return False
    # 记录当前请求时间
    request_times.append(current_time)
    return True


def load_classes(labels_path):
    with open(labels_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes


@app.route('/predict', methods=['POST'])
def predict():
    # 检查 API KEY
    # 从查询字符串中获取 API 密钥
    api_key = request.args.get('key')
    if args.key is None or api_key != args.key:
        return jsonify({'error': 'Unauthorized'}), 401

    # 检查速率限制
    if rate_limit and not check_rate_limit():
        return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429

    # 直接从请求内容中读取图像数据
    if request.content_type.startswith('image'):
        img_bytes = request.data
        np_img = np.frombuffer(img_bytes, np.uint8)
        origin_img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    elif 'image' in request.files:
        # 从文件上传中获取图像
        file = request.files['image']
        origin_img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    elif 'image' in request.json:
        # 从JSON中获取Base64编码的图像
        image_data = request.json['image']
        img_bytes = base64.b64decode(image_data)
        np_img = np.frombuffer(img_bytes, np.uint8)
        origin_img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    else:
        return jsonify({'error': 'No image data provided'}), 400

    if origin_img is None:
        return jsonify({'error': 'Invalid image format'}), 400

    start_inference_time = time.time()  # 开始推理计时

    img, ratio = preproc(origin_img, input_shape)
    ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
    output = session.run(None, ort_inputs)
    predictions = demo_postprocess(output[0], input_shape)[0]

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
    boxes_xyxy /= ratio
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=args.score_thr)

    # 判断是否需要可视化
    query_vis = request.args.get('vis', '0') == '1'
    save_vis = True if args.output_dir else Flask
    need_vis = query_vis or save_vis
    vis_base64 = None

    result = []
    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        for box, score, cls_ind in zip(final_boxes, final_scores, final_cls_inds):
            result.append({
                'box': box.tolist(),
                'score': score.item(),
                'class_id': int(cls_ind),
                'class_name': COCO_CLASSES[int(cls_ind)],
            })

        if need_vis:
            origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                        conf=args.score_thr, class_names=COCO_CLASSES)

        # 需要保存
        if save_vis:
            # 使用当前时间戳和随机数生成文件名
            timestamp = int(time.time())
            random_num = random.randint(1000, 9999)
            output_filename = f"{timestamp}_{random_num}.jpg"
            output_path = os.path.join(args.output_dir, output_filename)
            cv2.imwrite(output_path, origin_img)

        # 需要返回
        if query_vis:
            _, buffer = cv2.imencode('.jpg', origin_img)
            vis_base64 = base64.b64encode(buffer).decode('utf-8')

    inference_time = time.time() - start_inference_time  # 结束推理计时

    return jsonify({
        'data': result,
        'vis': vis_base64,
        'et': inference_time * 1000
    }), 200


if __name__ == '__main__':
    # 参数解析
    args = make_parser().parse_args()
    rate_limit = args.rate_limit
    mkdir(args.output_dir)

    # 加载模型和分类
    console_log('Loading model...')
    input_shape = tuple(map(int, args.input_shape.split(',')))
    session = onnxruntime.InferenceSession(args.model)
    COCO_CLASSES = load_classes(args.labels)

    # 停止计时
    takes_ms = int((time.time() - start_time) * 1000)
    takes_sec = round(takes_ms / 1000, 3)
    console_log('Server being started ('+str(takes_sec) + 's)! Listening on port ' + str(args.port))
    del start_time, takes_ms, takes_sec

    from waitress import serve
    serve(app, host='0.0.0.0', port=args.port)
    # app.run(host='0.0.0.0', port=args.port)

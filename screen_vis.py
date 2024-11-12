import argparse
import base64
import os
import random
import time

# 开始计时
start_time = time.time()
print('https://github.com/nkxingxh/yolox-onnx-api-server')
print('Loading libraries, please wait...')

import cv2
import numpy as np
import torch        # 解决 onnxruntime 找不到 cuda 的问题
import onnxruntime
from mss import mss
from PIL import Image, ImageDraw, ImageFont
from utils import mkdir, multiclass_nms, demo_postprocess , vis

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
    parser.add_argument("-m", "--model", type=str, required=True, help="指定ONNX模型文件。")
    parser.add_argument("-l", "--labels", type=str, required=True, help="分类标签文件。")
    parser.add_argument("-o", "--output_dir", type=str, default=None, help="可视化图片输出目录。为空则不保存可视化结果")
    parser.add_argument("-s", "--score_thr", type=float, default=0.3, help="全局置信度阈值。")
    parser.add_argument("-i", "--input_shape", type=str, default="640,640", help="指定推理的输入形状。")
    return parser

def load_classes(labels_path):
    with open(labels_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

if __name__ == '__main__':
    args = make_parser().parse_args()
    if args.output_dir:
        mkdir(args.output_dir)
    console_log('Loading model...')
    input_shape = tuple(map(int, args.input_shape.split(',')))
    session = onnxruntime.InferenceSession(
        args.model,
        # 'TensorrtExecutionProvider'
        providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    COCO_CLASSES = load_classes(args.labels)

    sct = mss()  # 初始化屏幕捕获
    monitor = sct.monitors[1]  # 全屏捕获

    while True:
        # 捕获屏幕
        screen_img = np.array(sct.grab(monitor))
        origin_img = cv2.cvtColor(screen_img, cv2.COLOR_BGRA2BGR)

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

        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                             conf=args.score_thr, class_names=COCO_CLASSES)

        # 将OpenCV图像转换为Pillow图像
        pil_image = Image.fromarray(cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        # 如果检测到目标则绘制检测框
        if dets is not None:
            for box, score, cls_ind in zip(final_boxes, final_scores, final_cls_inds):
                class_name = COCO_CLASSES[int(cls_ind)]
                draw.rectangle(
                    [(box[0], box[1]), (box[2], box[3])],
                    outline="red",
                    width=3
                )
                draw.text((box[0], box[1] - 10), f"{class_name} {score:.2f}", fill="red")

        # 将Pillow图像直接显示到屏幕
        pil_image.show()

        # 检查是否需要保存
        if args.output_dir:
            timestamp = int(time.time())
            random_num = random.randint(1000, 9999)
            output_filename = f"{timestamp}_{random_num}.jpg"
            output_path = os.path.join(args.output_dir, output_filename)
            pil_image.save(output_path)

        time.sleep(0.03)  # 控制检测刷新率

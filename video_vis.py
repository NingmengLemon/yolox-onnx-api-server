import argparse
import base64
import os
import random
import time

print('Loading libraries, please wait...')
import torch
import cv2
import numpy as np
import onnxruntime
from utils import mkdir, multiclass_nms, demo_postprocess, vis


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


def load_classes(labels_path):
    with open(labels_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes


def make_parser():
    parser = argparse.ArgumentParser("yolox-onnx-video-inference")
    parser.add_argument("-m", "--model", type=str, required=True, help="指定ONNX模型文件。")
    parser.add_argument("-l", "--labels", type=str, required=True, help="分类标签文件。")
    parser.add_argument("-v", "--video_path", type=str, required=True, help="指定输入视频文件路径。")
    parser.add_argument("-o", "--output_video", type=str, default=None, help="可指定输出视频文件路径。为空则不保存")

    parser.add_argument("-s", "--score_thr", type=float, default=0.65, help="全局置信度阈值。")
    parser.add_argument("-i", "--input_shape", type=str, default="640,640", help="指定推理的输入形状。")
    
    parser.add_argument("--tensorrt", action='store_true', help="启用TensorRT支持 (优先于CUDA)")
    parser.add_argument("--cuda", action='store_true', help="启用CUDA支持")
    return parser


def process_frame(frame, input_shape, session, score_thr, classes, output_dir=None, vis_dpi=1):
    img, ratio = preproc(frame, input_shape)
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
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=score_thr)

    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        vis_frame = vis(frame, final_boxes, final_scores, final_cls_inds,
                        conf=score_thr, class_names=classes, dpi=vis_dpi)

        if output_dir:
            timestamp = int(time.time())
            random_num = random.randint(1000, 9999)
            output_filename = f"{timestamp}_{random_num}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, vis_frame)

        return vis_frame
    return frame


if __name__ == '__main__':
    args = make_parser().parse_args()

    # providers
    exec_providers = []
    if args.tensorrt:
        exec_providers.append('TensorrtExecutionProvider')
    if args.cuda:
        exec_providers.append('CUDAExecutionProvider')
    exec_providers.append('CPUExecutionProvider')

    print('Loading model...')
    session = onnxruntime.InferenceSession(args.model, providers=exec_providers)
    classes = load_classes(args.labels)
    input_shape = tuple(map(int, args.input_shape.split(',')))

    print('Loading video I/O...')
    cap = cv2.VideoCapture(args.video_path)

    # 获取视频的宽、高和帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 创建 VideoWriter
    if args.output_video:
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))

    print('waiting for cap...')
    while cap.isOpened():
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame, input_shape, session, args.score_thr, classes, 
                                        vis_dpi=max(1, max(width, height) / 480))

        # 写入处理后的帧到输出视频
        if args.output_video:
            out.write(processed_frame)

        # 计算并显示帧率
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow("Video Inference", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if args.output_video:
        out.release()  # 释放VideoWriter
    cv2.destroyAllWindows()

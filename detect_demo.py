from __future__ import division

# from models import *
from models_rect import *

from utils.utils import *
# import utils.utils2 as u2

from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from torch2trt import TRTModule
from torchstat import stat
from torch2trt import torch2trt
from pathlib import Path


def Vedio_show(pred, last_img0, names):
    '''
    画图+显示视频用
    '''
    colors = [[0, 0, 255], [0, 255, 0]]
    for i, det in enumerate(pred):  # detections per image
        p, s, im0 = path, '', last_img0
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(
                img.shape[1:], det[:, :4], im0.shape).round()
            # 画框
            for *xyxy, conf, cls in det:
                label = '%s %.2f' % (names[int(cls)], conf)
                plot_one_box(xyxy, im0, label=label,
                             color=colors[int(cls)], line_thickness=1)
    # 画上FPS
    label = "%.2f" % (1/(time.time() - t0))+" FPS"
    # print(label)
    cv2.putText(im0, label, (10, 30), 0, 1, [
                0, 0, 0], thickness=3, lineType=cv2.LINE_AA)
    cv2.imshow(p, im0)
    if cv2.waitKey(1) == ord('q'):  # q to quit
        raise StopIteration


if __name__ == '__main__':
    '''
    本文件主要给出TensorRT模型的运行Demo，
    - TensorRT 转换的时候需要指定输入的形状，也就是说，输入对象的形状是固定的，
        如果开启矩形推理，那么就需要匹配该形状的TensorRT出现，具体计算方式就是图像每帧等比例缩放，
        再短边补充到32的倍数，这就是需要的TensorRT的形状了。
        - 一个比较偏的解决办法，就是合成较多输入形状的TensorRT，到时候根据视频的形状自动加载对应的TensorRT即可，
            这种办法需要预先判断视频形状。
    - TensorRT 需要再指定的机器中转化，如：如果运行在Jetson Nano上，则需要再Jetson Nano上转化。
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str,
                        default="config/prune_0.9_keep_0.01_20_shortcuts.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str,
                        default="/home/nano/Desktop/JetsonNano_Detected_Module/weights/trt_module_256x416.pth", help="path to weights file")
    # 控制半精度
    parser.add_argument("--half", default=True,
                        help="half precision FP16 inference")
    # 要检测的文件目录
    # input file/folder, 0 for webcam
    parser.add_argument(
        "--source", type=str, default="/home/nano/Desktop/JetsonNano_Detected_Module/data/静态.avi", help="source")
    # 控制矩形推理是否开启
    parser.add_argument("--rect", default=True, help="rectangular inference")
    # 控制是否使用TensorRT加速
    parser.add_argument("--tensorrt", default=True, help="use Tensorrt engine")
    parser.add_argument("--class_path", type=str,
                        default="data/helmet.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float,
                        default=0.5, help="object confidence threshold")
    parser.add_argument("--iou_thres", type=float, default=0.5,
                        help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int,
                        default=1, help="size of the batches")
    parser.add_argument("--device", type=bool, default=True,
                        help="use False:cpu or True:gpu")
    parser.add_argument("--img_size", type=int, default=416,
                        help="size of each image dimension")
    parser.add_argument("--view_img", default=True,
                        help="show img")
    opt = parser.parse_args(args=[])
    print(opt)

    # 选择设备
    device = torch.device(
        "cuda" if torch.cuda.is_available() and opt.device else "cpu")

    os.makedirs("output", exist_ok=True)

    TensorRT = opt.tensorrt
    print(TensorRT)

    # 加载模型，默认TensorRT模型，也可以选择非TensorRT
    if TensorRT is True:
        model_trt = TRTModule()
        model_trt.load_state_dict(torch.load(opt.weights_path))
    else:
        if opt.half:
            model = Darknet(opt.model_def, img_size=opt.img_size,
                            TensorRT=False, Half=True).to(device).half()
        else:
            model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
            print("not half")

        # 权重加载
        if opt.weights_path.endswith(".weights"):
            # Load darknet weights
            model.load_darknet_weights(opt.weights_path)
        else:
            # Load checkpoint weights
            model.load_state_dict(torch.load(opt.weights_path))
        # Set in evaluation mode 前向推理时候会忽略 BatchNormalization 和 Dropout
        model.eval()

    dataset = LoadImages(path=opt.source, img_size=(opt.img_size, opt.img_size), auto=opt.rect)
    names = load_classes(opt.class_path)  # Extracts class labels from file
    yolo_head = YOLOHead(config_path=opt.model_def)

    dete = None
    with torch.no_grad():
        t0 = time.time()
        path, img, im0s, vid_cap = next(dataset)
        pred = ''
        try:
            while vid_cap is not None:
                # 处理图片
                imputs = torch.from_numpy(img).to(device)
                imputs = imputs.half() if opt.half else imputs.float()  # uint8 to fp16/32
                imputs /= 255.0  # 0 - 255 to 0.0 - 1.0

                if imputs.ndimension() == 3:
                    imputs = imputs.unsqueeze(0)

                # 是否使用TensorRT引擎加速
                if TensorRT:
                    detections = yolo_head(model_trt(imputs))
                    # 数据预加载，主要是为了处理器并行，减少时间，串行时间大约在100ms-120ms左右
                    last_img0 = im0s
                    path, img, im0s, vid_cap = next(dataset)
                    # 这里主要执行上一帧的NMS
                    if dete is not None:
                        pred = non_max_suppression_new(dete, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, multi_cls=False,
                                                       classes=None, agnostic=False)
                    if pred and opt.view_img:
                        Vedio_show(pred, last_img0, names)
                        # 本帧的数据缓存，在下一帧检测的时候处理
                    dete = detections
                else:
                    # 由于这里没有做后续优化，因此就是数据加载和forward+NMS并行
                    detections = model(imputs)
                    path, img, im0s, vid_cap = next(dataset)
                    pred = non_max_suppression_new(detections, 0.15, 0.5, multi_cls=False,
                                                   classes=None, agnostic=False)
                    if opt.view_img:
                        Vedio_show(pred, img0s, names)
                # 输出FPS
                print("%.2f" % (1/(time.time() - t0))+" FPS")
                t0 = time.time()
        except Exception as e:
            print(e)

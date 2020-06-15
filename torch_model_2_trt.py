from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import argparse
import torch
from torch.autograd import Variable
from torchstat import stat
from torch2trt import torch2trt
from torch2trt import TRTModule


if __name__ == '__main__':
    '''
    本文件主演演示了一个pytorch的模型如何转换成tensorRT模型
    '''
    parser = argparse.ArgumentParser()
    # 模型config文件地址
    parser.add_argument("--model_def", type=str, default="/home/nano/Desktop/YOLOv3-Torch2TRT/config/prune_0.9_keep_0.01_20_shortcuts.cfg", help="path to model definition file")
    # 权重地址
    parser.add_argument("--weights_path", type=str, default="/home/nano/Desktop/YOLOv3-Torch2TRT/weights/prune_0.9_keep_0.01_20_shortcuts_finetune2_best.weights", help="path to weights file")
    # 控制半精度，Nano只支持FP16，不支持int8，因此不给int8选项
    parser.add_argument("--half",type = bool,default = True,help="half precision FP16 inference")
    # 控制使用GPU还是CPU，因为Nano只有一块128核的Maxwell，所以GPU用默认就行，不需要像训练服务器一样选择块号，
    parser.add_argument("--device", type=bool, default=True, help="use False:cpu or True:gpu")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    # 控制转换的形状，默认是正方形，如果需要转换成矩形，请输入相应的形状，由于YOLO特征层的特点，输入形状的每个值都必须是32的倍数
    parser.add_argument("--img_size", type=int, default=(256,416), help="size of each image dimension")
    # 模型保存名字，默认路径是：./weights/opt.trt_module.pth,转化后的某型类型是TRTModule，内含tensorRT engine
    parser.add_argument("--model_save_name",type = str,default = "trt_module_256x416.pth")
    opt = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and opt.device else "cpu")
    print(device)
    # 没有就创建一个
    os.makedirs("weights", exist_ok=True)
    # 模型选择
    # TensorRT 只能加速 YOLO 的特征提取网络, YOLO 层目前还不能应用，因此在使用的时候需要自定义YOLO Head（包含三层YOLO层）
    
    # 这段是模型转换
    model_trt = TRTModule()
    if opt.half:
        # 加载模型
        model_backbone = Darknet_Backbone(opt.model_def, img_size=opt.img_size).to(device).half()
    else:
        model_backbone = Darknet_Backbone(opt.model_def, img_size=opt.img_size).to(device)
        
    # torch 中.pth和.weights有所不同
    if opt.weights_path.split(".")[-1]=="pth":
        pass
        # load .pth文件
        model_backbone.load_state_dict(torch.load(opt.weights_path))
    else:
        # load .weights文件
        model_backbone.load_darknet_weights(opt.weights_path)
    model_backbone.eval()

    if opt.half:
        # 设置形状，x的内容不重要，主要是告诉转化器生成的输入是什么个形状的，3 = RGB 3通道
        x = torch.rand(size=(opt.batch_size, 3,  opt.img_size[0],opt.img_size[1])).to(device).half()
        # 这里开始模型的转换，fp16_mode=True表示开启半精度，源码中的int8选项，根据原作者的描述，并没有调试好
        model_trt = torch2trt(model_backbone, [x], fp16_mode=True)
    else:
        x = torch.rand(size=(opt.batch_size, 3,  opt.img_size[0],opt.img_size[1])).to(device)
        model_trt = torch2trt(model_backbone, [x])
    # 序列化保存模型
    torch.save(model_trt.state_dict(), "weights/{}".format(opt.model_save_name))
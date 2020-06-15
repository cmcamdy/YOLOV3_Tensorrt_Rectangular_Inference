from __future__ import division

import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from utils.parse_config import *
from utils.utils import build_targets, to_cpu, non_max_suppression

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import datetime
from torch2trt import torch2trt


def create_modules(module_defs, TensorRT):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    for module_i, module_def in enumerate(module_defs):

        modules = nn.Sequential()

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)

        elif module_def["type"] == "upsample":
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f"route_{module_i}", EmptyLayer())

        elif module_def["type"] == "shortcut":
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", EmptyLayer())

        elif module_def["type"] == "yolo":
            if TensorRT:
                pass
            else:
                anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
                # print(anchor_idxs)
                # Extract anchors
                anchors = [int(x) for x in module_def["anchors"].split(",")]
                anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
                anchors = [anchors[i] for i in anchor_idxs]
                # print(anchors)
                num_classes = int(module_def["classes"])
                img_size = int(hyperparams["height"])

                # Define detection layer
                yolo_layer = YOLOLayer(anchors, num_classes, img_size)
                modules.add_module(f"yolo_{module_i}", yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        # print("F.interpolate")
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0  # grid size

        # 添加以下compute_grid_offset类函数中的类变量
        self.stride = 0
        self.grid_x = 0
        self.grid_y = 0
        self.scaled_anchors = 0
        self.anchor_w = 0
        self.anchor_h = 0

    def compute_grid_offsets(self, grid_size, img_dim, cuda=True, Half=False):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        FloatTensor = torch.cuda.HalfTensor if Half else torch.cuda.FloatTensor
        self.img_dim = img_dim
        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None, Half=False):

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        FloatTensor = torch.cuda.HalfTensor if x.type() == "torch.cuda.HalfTensor" else torch.cuda.FloatTensor
        # LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        # ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        # 注释说明
        # x 是最后一层卷积输出的特征图，在输入图片大小为416×416的前提下
        # x[0],x[1],x[2],x[3] = batch size, 255, 13, 13
        # x[0],x[1],x[2],x[3] = batch size, 255, 26, 26
        # 255 = 3*(4+1+80)  3：我认为是mask的数量，也即每个cell生成的检测框数; 4:检测框坐标; 1:检测框置信度；80:类别数。
        # 检测框具体顺序为 Center x，Center y，Width，Height
        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)

        # 注释说明
        # prediction 的维度为 batch_size, num_anchors=3, grid_size, grid_size, num_classes + 5(coco:85)
        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
                .permute(0, 1, 3, 4, 2)  # permute: 将维度换位
                .contiguous()
        )
        # print(prediction.size())

        # 注释说明
        # Center x，Center y，Conf，Cls pred 用sigmoid函数限定其范围在0-1范围内
        # 为什么 w，h 不用限定范围？确实存在 w,h 大于1的是数据
        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf （检测框置信度）
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.
        # print(torch.max(w))
        # print(h)

        # 调试
        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, img_dim, cuda=x.is_cuda, Half=Half)

        # 注释说明
        # pred_box 表示网络预测的框
        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h
        # print(pred_boxes[..., 2].type())

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )
        # print(output.size())

        # 注释说明
        # target 用来表明是否是训练还是推理
        if targets is None:
            return output, 0
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            # 注释说明
            # loss_conf 正负样本带有各自权重（obj_scale，noobj_scale）
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }

            return output, total_loss


class Darknet_Backbone(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path, img_size=416, TensorRT=False, Half=False):
        super(Darknet_Backbone, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs, TensorRT)
        # self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")]
        self.img_size = img_size
        self.seen = 0
        # self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int8)

    def forward(self, x, targets=None):
        img_dim = x.shape[2]
        loss = 0
        layer_outputs, yolo_outputs = [], []
        last_convs = []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                # print(i, module_def["type"])
                x = module(x)
            elif module_def["type"] == "route":
                # print(i, module_def["type"])
                # print(module_def["layers"].split(","))
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            # 注释说明
            # shortcut 为YOLOv3的结构
            elif module_def["type"] == "shortcut":
                # print(i, module_def["type"])
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                # print(i, module_def["type"])
                last_convs.append(x)
            layer_outputs.append(x)

        return last_convs

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            # header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w


class YOLOHead(nn.Module):
    """
   Build a detection head.
    """

    def __init__(self, config_path, img_size=416):
        print("inital...........")
        super(YOLOHead, self).__init__()
        self.img_size = img_size
        self.module_defs = parse_model_config(config_path)
        self.yolo_layer = self.build()

    def build(self):
        # 构建YOLO层

        # 为了测试调用的而写,大约需要花费50ms
        print("building...........")

        hyperparams = self.module_defs.pop(0)
        img_size = int(hyperparams["height"])
        yolo_layer = []
        for module_i, module_def in enumerate(self.module_defs):
            if module_def["type"] == "yolo":
                anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
                # Extract anchors
                anchors = [int(x) for x in module_def["anchors"].split(",")]
                anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
                anchors = [anchors[i] for i in anchor_idxs]
                num_classes = int(module_def["classes"])
                # Define detection layer
                yolo_layer.append(YOLOLayer(anchors=anchors, num_classes=num_classes, img_dim=img_size))
        return yolo_layer

    def forward(self, backbone_out):
        print("forwarding...........")
        prevtime = time.time()
        if len(self.yolo_layer) == 2:
            x1, loss1 = self.yolo_layer[0](backbone_out[0], targets=None, img_dim=self.img_size)
            x2, loss2 = self.yolo_layer[1](backbone_out[1], targets=None, img_dim=self.img_size)
            yolo_out = to_cpu(torch.cat((x1, x2), 1))
            print("\t+ forward:Time:%s" % (datetime.timedelta(seconds=time.time() - prevtime)))
            return yolo_out
        else:
            start = time.time()
            x1, loss1 = self.yolo_layer[0](backbone_out[0], targets=None, img_dim=self.img_size)
            print("\t+ loss1 Time: %s" % (str(time.time() - start)))
            start = time.time()
            x2, loss2 = self.yolo_layer[1](backbone_out[1], targets=None, img_dim=self.img_size)
            print("\t+ loss2 Time: %s" % (str(time.time() - start)))
            start = time.time()
            x3, loss3 = self.yolo_layer[2](backbone_out[2], targets=None, img_dim=self.img_size)
            print("\t+ loss3 Time: %s" % (str(time.time() - start)))
            start = time.time()
            yolo_out = to_cpu(torch.cat((x1, x2, x3), 1))
            print("\t+ to_cpu Time: %s" % (str(time.time() - start)))
            print("\t+ forward:Time:%s" % (datetime.timedelta(seconds=time.time() - prevtime)))
            return yolo_out


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path, img_size=416, TensorRT=False, Half=False):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs, TensorRT)
        # if Half is False:
        #     self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")]
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x, targets=None):
        # print(x.type())
        if x.type() == "torch.cuda.HalfTensor":
            Half = True
        else:
            Half = False
        img_dim = x.shape[2]
        loss = 0
        layer_outputs, yolo_outputs = [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                # print(i, module_def["type"])
                x = module(x)
            elif module_def["type"] == "route":
                # print(i, module_def["type"])
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            # 注释说明
            # shortcut 为YOLOv3的结构
            elif module_def["type"] == "shortcut":
                # print(i, module_def["type"])
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                # print(i, module_def["type"])
                x, layer_loss = module[0](x, targets, img_dim, Half)
                loss += layer_loss
                yolo_outputs.append(x)
            layer_outputs.append(x)
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
        # print(yolo_outputs.size())
        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()

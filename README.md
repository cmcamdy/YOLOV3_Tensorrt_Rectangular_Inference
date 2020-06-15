模型部署源代码（Jetson Nano）说明



[toc]

## 项目结构图
> 模型部署源代码（Jetson Nano）
>   │  detect_demo.py					：演示demo
>   │  models.py							 ：只支持方形推理的模型	
>   │  models_rect.py					 ：支持方形推理和矩形推理的模型						
>   │  README.md						：说明文档
>   │  torch_model_2_trt.py			：模型转换demo
>   │
>   ├─config
>   │  请下载附件获取模型结构文件（cfg）
>   │
>   ├─data									
>   │  │  helmet.names					：类别，hat,person
>   │  │
>   │  ├─custom
>   │  │  ├─images						  ：保存图片
>   │  │  └─labels                 			：保存标签
>   │  ├─output								：保存检测结果
>   │  └─samples  							：保存测试原数据
>   ├─utils
>   │      datasets.py						：进行数据加载
>   │      parse_config.py				 ：解析cfg
>   │      utils.py							   ：工具包
>   └─weights								 ：保存权重文件,请下载附件获取模型权重文件

## 代码使用说明

- 1.准备好YOLOv3模型以及相应cfg
- 2.执行 torch_model_2_trt.py，模型转换，注意，转换TensorRT的时候需要设置X的形状，如：[1,3,416,256]，后两位表示输入形状。
- 3.执行detect_demo.py ，进行模型推理，注意，输入图片形状必须是TensorRT对应的形状，否则TensorRT模型的输出是无效的。

## 代码文件说明

- 该仓库主要代码分为两大部分
    - 1.模型转换Demo
    - 2.模型推理Demo
### 模型转换Demo
- 文件名：torch_model_2_trt.py
- 概述：该文件功能是将pytorch的模型转换成包含TensorRT加速引擎的TRTModule模型，在GPU层面上实现加速效果。
- 主要参数说明：
    - **model_def**：模型config文件地址
    
    - **weights_path**：权重地址，权重可能有两种后缀，分别为.weights和.pth，无论那种，cfg必须是对应的
    
    - **half**：半精度
    
    - **device**：Jetson Nano拥有一块128core的MaxwellGPU，因此选GPU时不需要选块号，以T/F表示GPU/CPU
    
    - **img_size**:模型需要定义输入形状，默认是正方形，如果需要转换成矩形，请输入相应的形状，由于YOLO特征层的特点，输入形状的每个值都必须是32的倍数。
    
    	**注意：如果输入形状不对应，检测会失败**
    
    - **model_save_name**：转换好的模型序列化保存需要用到的名字，建议将输入图像尺寸作为名字的一部分。

- 输出：转换、序列化后的模型文件。

### 模型演示Demo

- 文件名：detect_demo.py
- 概述：该文件功能是演示在Nano上从：读取视频——>推理——>输出的整个过程。主体思想是：利用CPU和GPU独立运行的特性使得三大步骤并行执行
- 主要参数说明：
	- model_def、weights_path、half参考上文
	- rect：控制矩形推理是否执行
	- tensorrt：控制是否使用TensorRT模型
	- conf_thres：置信度阈值
	- iou_thres：iou阈值
	- batch_size：每个batch的大小，默认1
	- img_size：长边大小416，若输入是不是正方形，则长边为416，短边为等比例缩小后最小的32的倍数
	- view_img：是否输出检测结果图像
- 输出：控制台输出帧率（视频左上角也有），检测结果视频播放

## 部分加速原理说明
### 关于本项目矩形推理加速的原理说明

- 具体流程可以参考下图

- ![](https://img-blog.csdnimg.cn/20200418111341728.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3N5bXVhbXVh,size_16,color_FFFFFF,t_70)

- 矩形推理发生的地方位于两个阶段,代码地址分别是:utils.datasets.py和models_rect.py中：
	- 1、LoadImages：```dataset = LoadImages(path=opt.source, img_size=(opt.img_size, opt.img_size), auto=opt.rect)```
	- 2、YOLOLayer的forward：```yolo_head = YOLOHead(config_path=opt.model_def)```
		- 本项目预留了原版models，不支持矩形推理，可与新版models_rect.py对比查看

### 本项目数据预加载加速原理说明

- 利用CPU与GPU可以并行的特点，将下一帧的数据加载与本帧模型推理放在同一时间处理，此时耗时较短的数据加载便会在宏观上消失，具体参考detect_demo.py中的while vid_cap部分代码。
	- 更新：后处理NMS也加入进来，目前是:（上一帧的NMS+下一帧的数据加载）与（本帧的模型推理）并行。



## Jetson Nano上效果对比

- Nano的算力并不算强，Maxwell这块U算力大约是1060的$$\frac{1}{7}  - \frac{1}{10}$$
- 目前做过如下测试：

| 设备型号    | 运行模型   | GPU  | TensorRT加速 | 剪枝 | 矩形推理 | FPS  |
| ----------- | ---------- | ---- | ------------ | ---- | -------- | ---- |
| GTX1060     | YOLOV3-SPP | √    |              |      |          | 18.4 |
| Jetson Nano | YOLOV3-SPP |      |              |      |          | 0.3  |
| Jetson Nano | YOLOV3-SPP | √    |              |      |          | 2    |
| Jetson Nano | YOLOV3-SPP | √    | √            |      |          | 3.7  |
| Jetson Nano | YOLOV3-SPP | √    | √            | √    |          | 12.3 |
| Jetson Nano | YOLOV3-SPP | √    | √            | √    | √        | 19.2 |


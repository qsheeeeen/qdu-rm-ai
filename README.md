# 青岛大学 RoboMaster 视觉 人工智能 代码开源

UNFINISHED & UNTESTED.

软件正处在开发初期，只完成了视觉的核心部分，其余部分正在开发中。使用后果自负。

## 软件功能介绍

本开源软件为青岛大学未来战队机器人的视觉和人工智能的代码。参考了其他战队代码和各种开源机器人项目，从零编写而成。中心思想：

- 使用基于pytorch的yolov5算法，训练得到的权重和模型导出到ONNX格式。
- 在妙算平台使用TensorRT运行导出的模型。
- 一个项目适配不同不通型号的机器人。

这样做增加代码了的重用，减少了工作量。实现了可以通过DLA（深度学习加速器）加速妙算上模型的推断速度，切换所适配的机器人和操作手。代码完成后只需要烧写一次，减少维护的工作量，减少出错的可能性。

## 图片展示

### 目标识别效果

![目标识别效果](./image/test_best.jpg?raw=true "目标识别效果")

### TODO：TensorRT加速效果

## 依赖&环境

- 依赖：OpenCV、BehavoirTree.CPP、MVS SDK、spdlog、CUDA、TensorRT。
- Linux平台，使用CMake和VS Code开发。未在Windows平台测试。

## 使用说明

- 安装依赖

  - [OpenCV](https://docs.opencv.org/4.4.0/d7/d9f/tutorial_linux_install.html)

  - [BehavoirTree.CPP](https://github.com/BehaviorTree/BehaviorTree.CPP).

  - MVS SDK from [HIKROBOT](https://www.hikrobotics.com/service/soft.htm). Follow the instructions in INSTALL.

  - [spdlog](https://github.com/gabime/spdlog).

  - [CUDA](https://developer.nvidia.com/cuda-downloads)

  - [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)
  
  - Run `ldconfig` after install BehaviorTreeV3

- 训练神经网络

  ```sh

  # 以下脚本涉及相对路径，需要在本文件夹内运行。
  cd ./utils

  # 处理数据集
  sh python3 roco2x.py ----dji-roco-dir=/path/to/DJI ROCO/

  # 训练模型
  sh ./train_vision.sh
  ```

- 编译代码.

  ```sh
  git clone --recursive https://github.com/qsheeeeen/qdu-robomaster-ai
  cd qdu-robomaster-ai
  mkdir build
  cd build
  cmake ..
  make -j
  ```

- 运行
  - 暂时：使用VS Code运行，或者在repo目录里运行build里的二进制文件

  - TODO: 
  ```sh
  #安装后直接使用
  qdu-rm-ai-auto-aim
  qdu-rm-ai-full-auto
  qdu-rm-ai-radar
  ```

## 文件目录结构及文件用途说明

| 文件夹 | 内容 |
| ---- | ----  |
| image | 图片。包含效果展示，程序运行时也会往里存图片 |
| logs | 程序运行日志 |
| mid | 程序的中间产物，包含训练好的权重，TensorRT的engine等 |
| src | 源代码 |
| test | 测试代码 |
| third_party | 第三方软件 |
| utils | 辅助的脚本和文件 |

| src内 | 内容 |
| ---- | ----  |
| app | 应用。包含哨兵使用的全自动、步兵等使用的自瞄和雷达 |
| behavior | 行为库。基于行为树开发的AI |
| device | 设备库。外接设备的抽象 |
| vision | 视觉库。目标识别等代码 |

## 系统介绍

### 软件流程图

![步兵嵌入式硬件框图](./Image/嵌入式程序流程图.png?raw=true "步兵嵌入式硬件框图")

### 软件层级图

![嵌入式程序结构图](./Image/嵌入式程序结构图.png?raw=true "嵌入式程序结构图")

## TODO

- Debug Detector 输出。
- 完成输入预处理
- Add logging and file on top of LittleFS.
- Add gimbal soft limit.
- 给BSP想个更好的名字，备选"mcu"。
- 给BSP USB print加保护，允许不同进程的使用。
  - 给所有BSP加保护
  - device.c里面加上一个Device_Init()，在里面初始化所有mutex
- CAN设备代码优化。消息解析发送方向。
  - CAN设备动态初始化，保存好几组配置。

## Roadmap

近期：

1. 添加与MCU的通信协议，完成联调。

1. 添加 CInstall

1. 添加 CTest

远期：

1. ONNX1.8发布后适配yolov5的master

1. 添加Int8运行

1. 添加自瞄部分代码。

1. 添加机关击打。

1. 【视情况】添加雷达部分代码。

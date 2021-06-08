# 青岛大学 RoboMaster 视觉 人工智能 代码开源

***Developing.***

[Gitee](https://gitee.com/qsheeeeen/qdu-rm-ai)
[Github](https://github.com/qsheeeeen/qdu-rm-ai)

软件正处在开发初期，只完成了视觉的核心部分，其余部分正在开发中。

## 软件介绍

本开源软件为青岛大学未来战队机器人的视觉和人工智能的代码。参考了其他战队代码和各种开源机器人项目，从零编写而成。中心思想：

- 基于OpenCV的识别算法
- 基于行为树设计哨兵的AI
- 一个项目适配不同型号的机器人。

这样做增加代码了的重用，减少了工作量。实现了通过DLA(深度学习加速器)加速妙算上模型的推断速度。利用行为树实现了可控的复杂行为。

## 图片展示

| ![YOLO识别效果](./image/test_yolo.jpg?raw=true "YOLO识别效果") |
|:--:|
| *YOLO识别效果* |

| ![装甲板匹配效果](./image/test_origin.png?raw=true "装甲板匹配效果") |
|:--:|
| *本算法识别效果* |

| ![TensorRT加速效果](./image/compare.jpg?raw=true "TensorRT加速效果") |
|:--:|
| *TODO：TensorRT加速效果* |

## 依赖 & 环境

- 依赖
  - [OpenCV](https://docs.opencv.org/4.5.1/d7/d9f/tutorial_linux_install.html)
  - [BehavoirTree.CPP](https://github.com/BehaviorTree/BehaviorTree.CPP).
  - [MVS SDK from HIKROBOT](https://www.hikrobotics.com/service/download/0/0).
  - [spdlog](https://github.com/gabime/spdlog).
  - [Google Test](https://github.com/google/googletest)
  - [oneTBB](https://github.com/oneapi-src/oneTBB) or `libtbb-dev`
  - 可选
    - [CUDA](https://developer.nvidia.com/cuda-downloads)
    - [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)

- 开发测试环境
  - Ubuntu
  - WSL2(不能使用工业相机)

## 使用说明

1. 安装依赖
    1. 根据上面链接安装相关依赖
    1. 安装完成后运行`ldconfig`

1. 获得代码

    ```sh
    git clone --recursive https://github.com/qsheeeeen/qdu-rm-ai
    # or
    git clone --recursive https://gitee.com/qsheeeeen/qdu-rm-ai
    
    ```

1. 编译 & 安装

    ```sh
    cd qdu-rm-ai
    mkdir build
    cd build
    cmake ..
    make -j8 # 根据CPU选择合适的数字
    make install
    ```

1. 神经网络(可选)

    1. 准备

        ```sh
        # 安装本项目需要的Python模块。
        pip3 install -r qdu-rm-ai/requirements.txt

        # 安装YOLOv5需要的Python模块
        pip3 install -r qdu-rm-ai/third_party/yolov5/requirements.txt
        ```

    1. 训练

        ```sh
        # 以下脚本涉及相对路径，需要在此文件夹内运行。
        cd qdu-rm-ai/utils

        # 处理数据集
        python3 roco2x.py --dji-roco-dir=path/to/DJI ROCO/

        # 训练导出模型
        sh ./train_vision.sh
        ```

1. 运行

    ```sh
    cd qdu-rm-ai/runtime
    # 根据应用选择程序
    auto-aim # sentry / radar ...
    ```

## 文件目录结构及文件用途说明

| 文件夹 | 内容 | 备注 |
| ---- | ---- | ---- |
| image | 图片 | 包含效果展示、测设产物等 |
| runtime | 运行环境 | 包含运行所需文件，和运行过程产生的文件 |
| src | 源代码 |
| tests | 测试代码 |
| third_party | 第三方软件 |
| utils | 工具 | 脚本和文件 |

| src内 | 内容 | 备注 |
| ---- | ---- | ---- |
| app | 应用 | 包含哨兵程序、自瞄算法、雷达程序等 |
| behavior | 行为库 | 基于行为树开发的AI |
| nn | 神经网络库 | 基于神经网络的算法 |
| demo | 样例 | 演示用的例子 |
| device | 设备库 | 外接设备的抽象 |
| vision | 视觉库 | 目标识别等代码 |

## 系统介绍

### 软件流程图

| ![视觉程序框图](./image/视觉程序框图.png?raw=true "步兵嵌入式硬件框图") |
|:--:|
| *视觉程序框图* |

### 行为树演示

| ![行为树演示](./image/行为树演示.png?raw=true "行为树演示") |
|:--:|
| *行为树演示* |

## Roadmap

近期：

1. 实现类似多级流水线的视觉算法流程。[参考文章](https://opencv.org/hybrid-cv-dl-pipelines-with-opencv-4-4-g-api/)

远期：

1. 添加机关击打

1. 第二阶段
    1. 使用基于pytorch的yolov5算法，训练得到的权重和模型导出到ONNX格式。
    1. 在妙算平台使用TensorRT运行导出的模型。
    1. 添加Int8运行

1. 第三阶段
    1. 添加雷达部分代码

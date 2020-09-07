# qdu-robomaster-vision

## Setup

- Install dependency

  - [OpenCV](https://docs.opencv.org/4.4.0/d7/d9f/tutorial_linux_install.html)

  - [BehavoirTree.CPP](https://github.com/BehaviorTree/BehaviorTree.CPP).

  - MVS SDK from [HIKROBOT](https://www.hikrobotics.com/service/soft.htm). Follow the instructions in INSTALL.

  - [spdlog](https://github.com/gabime/spdlog).

  - [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)

  - [CUDA](https://developer.nvidia.com/cuda-downloads)
  
- Build code.

  ```cmd
  git clone https://github.com/qsheeeeen/qdu-robomaster-ai
  cd qdu-robomaster-ai
  mkdir build
  cd build
  cmake ..
  make -j
  ```

## Reminder

- Run `ldconfig` after install BehaviorTreeV3

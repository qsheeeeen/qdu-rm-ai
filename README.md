# qdu-robomaster-vision

## Setup

- Install dependency

  - Install [BehavoirTree.CPP](https://github.com/BehaviorTree/BehaviorTree.CPP). Follow the instructions in README.

  - Install MVS SDK from [HIKROBOT](https://www.hikrobotics.com/service/soft.htm). Follow the instructions in INSTALL.

  - Install MVS SDK from [spdlog](https://github.com/gabime/spdlog). Follow the instructions in README.

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

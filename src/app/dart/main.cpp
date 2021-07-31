#include "app.hpp"
#include "servo.hpp"

/* 飞镖行为状态 */
enum class DartStage {
  kREADY,   /* 准备状态 */
  kASCEND,  /* 上升状态 */
  kDESCEND, /* 下降状态 */
  kLAND     /* 着陆状态 */
};

class Dart : private App {
 private:
  uint32_t lask_wakeup;
  float dt;

  Servo servo;
  DartStage stage; /* 控制阶段 */

  void Dart_ReadyControl() { /* 读取气压计，计算高度 */
  }

  void Dart_AscendControl() {
    /* 根据目标能量控制飞镖推进力 */
    /* 不进行姿态控制，减少能量损耗 */
  }

  void Dart_DescendControl() {
    /* 读取摄像头数据 */
    /* 计算目标落点 */
    /* 计算目标能量 */
    /* 估计现有能量 */
    /*  */
  }

  void Dart_LandControl() {}

  void Dart_UpdateStage() {}

  void Control() {
    Dart_UpdateStage();

    /* 控制相关逻辑 */
    switch (stage) {
      case DartStage::kREADY:
        Dart_ReadyControl();
        break;

      case DartStage::kASCEND:
        Dart_AscendControl();
        break;

      case DartStage::kDESCEND:
        Dart_DescendControl();
        break;

      case DartStage::kLAND:
        Dart_LandControl();
        break;
    }
  }

  void Init();

  void UpdateFeedback();

 public:
  Dart(const std::string &log_path) : App(log_path) {
    SPDLOG_WARN("***** Setting Up Dart Control system. *****");

    /* 初始化设备 */
  }

  ~Dart() {
    /* 关闭设备 */

    SPDLOG_WARN("***** Shuted Down Dart Control system. *****");
  }

  /* 运行的主程序 */
  void Run() {
    SPDLOG_WARN("***** Running Dart Control system. *****");

    while (1) {
      SPDLOG_DEBUG("Tiking");
    }
  }
};

int main(int argc, char const *argv[]) {
  (void)argc;
  (void)argv;

  Dart dart("logs/dart.log");
  dart.Run();

  return EXIT_SUCCESS;
}

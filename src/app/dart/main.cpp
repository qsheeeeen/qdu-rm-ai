#include "app.hpp"
#include "servo.hpp"

class Dart : private App {
 private:
  Servo servo;

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

#pragma once

#include "app.hpp"
#include "servo.hpp"

class Dart : private App {
 private:
  Servo servo;

 public:
  Dart() {
    SPDLOG_WARN("***** Starting Dart Control system. *****");
    PrepareLogging("logs/dart.log");

    /* 初始化设备 */
  }

  ~Dart() {
    /* 关闭设备 */

    SPDLOG_WARN("***** Shuted Down Dart Control system. *****");
  }

  /* 运行的主程序 */
  void Run() {
    while (1) {
      SPDLOG_DEBUG("Tiking");
    }
  }
};

int main(int argc, char const *argv[]) {
  (void)argc;
  (void)argv;
  Dart dart;

  dart.Run();

  return EXIT_SUCCESS;
}

#include <iostream>

#include "behavior.hpp"
#include "camera.hpp"
#include "robot.hpp"
#include "spdlog/spdlog.h"
#include "vision.hpp"

int main(int argc, char const* argv[]) {
  (void)argc;
  (void)argv;

  spdlog::set_level(spdlog::level::trace);
  spdlog::warn("*****Running Auto Aim.*****");

  Robot bot("/home/qs/virtual_robot.txt");
  Camera cam(0);

  // Test only. TOOD: Remove.
  TestTree();
  TestOpenCV();
  TestVideoWrite();

  // Init behavior.
  // Run true tree.

  return 0;
}

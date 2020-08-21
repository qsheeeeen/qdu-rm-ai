#include <iostream>

#include "behavior.hpp"
#include "camera.hpp"
#include "robot.hpp"
#include "vision.hpp"

int main(int argc, char const* argv[]) {
  (void)argc;
  (void)argv;

  std::cout << "Auto aimer started." << std::endl;

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

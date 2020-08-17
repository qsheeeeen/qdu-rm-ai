#include <iostream>
#include <thread>

#include "behavior.hpp"
#include "camera.hpp"
#include "robot.hpp"
#include "vision.hpp"

void processing_robot(Robot& robot) {
  for (int i = 0; i < 5; ++i) {
    std::cout << "Robot thread executing.\n" << i << std::endl;
    // robot.Parse();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

int main(int argc, char const* argv[]) {
  (void)argc;
  (void)argv;

  std::cout << "Auto aimer started." << std::endl;

  Robot bot;
  std::thread robot_thread(processing_robot, std::ref(bot));
  // Test only. TOOD: Remove.

  // Camera cam(0);

  TestTree();
  TestOpenCV();
  TestVideoWrite();

  robot_thread.join();

  // Init behavior.
  // Run true tree.

  return 0;
}

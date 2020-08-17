#include <iostream>

#include "behavior.hpp"
#include "camera.hpp"
#include "vision.hpp"

int main(int argc, char const *argv[]) {
  (void)argc;
  (void)argv;
  std::cout << "Auto aimer started." << std::endl;

  // Test only. TOOD: Remove.
  auto cam = Camera(0);

  TestTree();
  TestOpenCV();
  TestVideoWrite();

  // Init behavior.
  // Run true tree.

  return 0;
}

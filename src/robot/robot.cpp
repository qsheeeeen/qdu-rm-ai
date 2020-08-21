#include "robot.hpp"

#include <fstream>
#include <iostream>

void Robot::WorkThread() {
  while (continue_parse_) {
    dev_.read(buff, sizeof(holder_t));
    Parse();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

void Robot::Parse() {}

Robot::Robot(const std::string &dev_path) {
  std::cout << "Create Robot." << std::endl;

  dev_.open(dev_path);
  if (!dev_.is_open()) throw std::runtime_error("Can't open device.");

  continue_parse_ = true;
  parse_thread_ = std::thread(&Robot::WorkThread, this);
}

Robot::~Robot() {
  dev_.close();
  std::cout << "Camera Destried." << std::endl;
}

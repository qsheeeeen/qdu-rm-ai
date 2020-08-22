#include "robot.hpp"

#include <iostream>

void Robot::ComThread() {
  while (continue_parse_) {
    dev_.read((char*)&status_, sizeof(recv_holder_t));

    commandq_mutex_.lock();
    if (!commandq_.empty()) {
      dev_.write((char*)&commandq_.front(), sizeof(recv_holder_t));
      commandq_.pop();
    }
    commandq_mutex_.unlock();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

void Robot::CommandThread() {}

Robot::Robot(const std::string &dev_path) {
  std::cout << "Create Robot." << std::endl;

  dev_.open(dev_path, std::ios::binary);
  if (!dev_.is_open()) throw std::runtime_error("Can't open Robot device.");

  continue_parse_ = true;
  parse_thread_ = std::thread(&Robot::ComThread, this);
}

Robot::~Robot() {
  dev_.close();
  std::cout << "Camera Destried." << std::endl;
}

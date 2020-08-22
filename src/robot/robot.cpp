#include "robot.hpp"

#include <iostream>

void Robot::WorkThread() {
  while (continue_parse_) {
    dev_.read(recv_buff_, sizeof(recv_holder_t));
    Parse();
    commandq_mutex.lock();
    if (!commandq_.empty()) {
      dev_.write((char*)&commandq_.front(), sizeof(recv_holder_t));
      commandq_.pop();
    }
    commandq_mutex.unlock();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

void ParseThread();

void Robot::Parse() {}

Robot::Robot(const std::string &dev_path) {
  std::cout << "Create Robot." << std::endl;

  dev_.open(dev_path, std::ios::binary);
  if (!dev_.is_open()) throw std::runtime_error("Can't open Robot device.");

  continue_parse_ = true;
  parse_thread_ = std::thread(&Robot::WorkThread, this);
}

Robot::~Robot() {
  dev_.close();
  std::cout << "Camera Destried." << std::endl;
}

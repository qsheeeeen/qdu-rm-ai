#include "robot.hpp"

#include "spdlog/spdlog.h"

void Robot::ComThread() {
  SPDLOG_DEBUG("[ComThread] Started.");

  while (continue_parse_) {
    serial_.Recv((char *)&status_, sizeof(RecvHolder));

    commandq_mutex_.lock();
    if (!commandq_.empty()) {
      serial_.Trans((char *)&commandq_.front(), sizeof(RecvHolder));
      commandq_.pop();
    }
    commandq_mutex_.unlock();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  SPDLOG_DEBUG("[ComThread] Stoped.");
}

void Robot::CommandThread() {
  SPDLOG_DEBUG("[CommandThread] Started.");

  SPDLOG_DEBUG("[CommandThread] Stoped.");
}

Robot::Robot(const std::string &dev_path) {
  serial_.Open(dev_path);
  serial_.Config();
  if (!serial_.IsOpen()) {
    SPDLOG_ERROR("Can't open device.");
  }

  continue_parse_ = true;
  parse_thread_ = std::thread(&Robot::ComThread, this);

  SPDLOG_DEBUG("Constructed.");
}

Robot::~Robot() {
  serial_.Close();

  continue_parse_ = false;
  parse_thread_.join();
  SPDLOG_DEBUG("Destructed.");
}

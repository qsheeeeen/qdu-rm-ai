#include "robot.hpp"

#include "spdlog/spdlog.h"

void Robot::ComThread() {
  spdlog::debug("[Robot][ComThread] Running.");

  while (continue_parse_) {
    dev_.read((char *)&status_, sizeof(recv_holder_t));

    commandq_mutex_.lock();
    if (!commandq_.empty()) {
      dev_.write((char *)&commandq_.front(), sizeof(recv_holder_t));
      commandq_.pop();
    }
    commandq_mutex_.unlock();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  spdlog::debug("[Robot][ComThread] Stoped.");
}

void Robot::CommandThread() {
  spdlog::debug("[Robot][CommandThread] Running.");

  spdlog::debug("[Robot][CommandThread] Stoped.");
}

Robot::Robot(const std::string &dev_path) {
  spdlog::debug("[Robot] Creating.");

  dev_.open(dev_path, std::ios::binary);
  if (!dev_.is_open()) {
    spdlog::error("[Robot] Can't open Robot device.");
    throw std::runtime_error("[Robot] Can't open Robot device.");
  }

  continue_parse_ = true;
  parse_thread_ = std::thread(&Robot::ComThread, this);

  spdlog::debug("[Robot] Created.");
}

Robot::~Robot() {
  spdlog::debug("[Robot] Destroying.");
  dev_.close();
  spdlog::debug("[Robot] Destried.");
}

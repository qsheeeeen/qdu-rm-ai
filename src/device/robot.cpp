#include "robot.hpp"

#include "spdlog/spdlog.h"

void Robot::ComThread() {
  spdlog::debug("[Robot][ComThread] Running.");

  while (continue_parse_) {
    com_.Recv((char *)&status_, sizeof(RecvHolder));

    commandq_mutex_.lock();
    if (!commandq_.empty()) {
      com_.Trans((char *)&commandq_.front(), sizeof(RecvHolder));
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

  com_.Open(dev_path);
  if (!com_.IsOpen()) {
    spdlog::error("[Robot] Can't open Robot device.");
    throw std::runtime_error("[Robot] Can't open Robot device.");
  }

  continue_parse_ = true;
  parse_thread_ = std::thread(&Robot::ComThread, this);

  spdlog::debug("[Robot] Created.");
}

Robot::~Robot() {
  spdlog::debug("[Robot] Destroying.");
  com_.Close();
  spdlog::debug("[Robot] Destried.");
}

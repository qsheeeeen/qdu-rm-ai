#pragma once

#include <mutex>
#include <queue>
#include <thread>

#include "crc16.hpp"
#include "protocol.h"
#include "serial.hpp"

class MCU {
 private:
  Serial serial_;
  bool continue_parse_ = false;
  std::thread parse_thread_;
  std::queue<Protocol_AI_t> commandq_;
  std::mutex commandq_mutex_;

  Protocol_Referee_t status_refe_;
  Protocol_MCU_t status_mcu_;

  void ComThread();
  void CommandThread();

 public:
  MCU(const std::string &dev_path);
  ~MCU();
  void Command();
};

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
  bool thread_continue = false;
  std::thread thread_recv_, thread_trans_;

  std::queue<Protocol_Data_AI_t> commandq_;
  Protocol_Data_Referee_t ref_;
  Protocol_Data_MCU_t mcu_;

  std::mutex mutex_commandq_, mutex_ref_, mutex_mcu_;

  void ThreadRecv();
  void ThreadTrans();

 public:
  MCU(const std::string &dev_path);
  ~MCU();
  void Command();
};

#pragma once

#include <mutex>
#include <queue>
#include <thread>

#include "serial.hpp"

typedef struct {
  double holder;
} RecvHolder;

typedef struct {
  double holder;
} CommandHolder;

class MCU {
 private:
  Serial serial_;
  bool continue_parse_ = false;
  std::thread parse_thread_;
  std::queue<CommandHolder> commandq_;
  std::mutex commandq_mutex_;

  RecvHolder status_;
  void ComThread();
  void CommandThread();

 public:
  MCU(const std::string &dev_path);
  ~MCU();
  void Command();
};

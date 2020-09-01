#pragma once

#include <mutex>
#include <queue>
#include <thread>

#include "serial.hpp"

typedef struct {
  float holder;
} RecvHolder;

typedef struct {
  float holder;
} CommandHolder;

class Robot {
 private:
  Serial com_;
  bool continue_parse_ = false;
  std::thread parse_thread_;
  std::queue<CommandHolder> commandq_;
  std::mutex commandq_mutex_;

  RecvHolder status_;
  void ComThread();
  void CommandThread();

 public:
  Robot(const std::string &dev_path);
  ~Robot();
  void Command();
};

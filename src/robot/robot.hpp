#pragma once

#include <fstream>
#include <mutex>
#include <queue>
#include <thread>

typedef struct {
  float holder;
} recv_holder_t;

typedef struct {
  float holder;
} command_holder_t;

class Robot {
 private:
  std::fstream dev_;
  bool continue_parse_ = false;
  std::thread parse_thread_;
  std::queue<command_holder_t> commandq_;
  std::mutex commandq_mutex;

  char recv_buff_[sizeof(recv_holder_t)];
  void WorkThread();
  void Parse();

 public:
  Robot(const std::string &dev_path);
  ~Robot();
  void Command();
};

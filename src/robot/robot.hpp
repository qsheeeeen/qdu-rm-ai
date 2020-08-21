#pragma once

#include <fstream>
#include <thread>

typedef struct {
  float holder;
} holder_t;

class Robot {
 private:
  std::fstream dev_;
  bool continue_parse_ = false;
  std::thread parse_thread_;

  char buff[sizeof(holder_t)];
  void WorkThread();
  void Parse();

 public:
  Robot(const std::string &dev_path);
  ~Robot();
  void Command();
};

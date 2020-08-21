#pragma once

#include <thread>

class Robot {
 private:
  std::fstream dev;
  bool continue_parse_ = false;
  std::thread parse_thread_;

  void WorkThread();
 public:
  Robot(std::string dev_path);
  ~Robot();

  bool Connect();
  bool Disconnect();

  bool Parse();
  bool Send();
};

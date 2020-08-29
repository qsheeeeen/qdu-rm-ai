#pragma once

#include <string>

enum class BaudRate {
  kBR9600,
  KBR115200,
};

class Serial {
 private:
  int dev_;

 public:
  Serial();
  Serial(const std::string &dev_path);
  ~Serial();
  void Open(const std::string &dev_path);
  bool IsOpen();
  bool Config(bool parity = false, bool stop_bit = false,
              bool flow_ctrl = false, BaudRate br = BaudRate::KBR115200);
  ssize_t Trans(const void* buff, size_t len);
  ssize_t Recv(void* buff, size_t len);
  int Close();
};

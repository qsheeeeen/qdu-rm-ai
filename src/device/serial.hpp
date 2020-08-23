#pragma once

#include <string>

typedef enum {
  kBR9600,
  KBR115200,
} BaudRate;

class Serial {
 private:
  int dev_;

 public:
  Serial();
  Serial(const std::string &dev_path);
  ~Serial();
  void Open(const std::string &dev_path);
  bool IsOpen();
  bool Config(bool parity, bool stop_bit, bool flow_ctrl, BaudRate br);
  ssize_t Trans(char buff[], int len);
  ssize_t Recv(char buff[], int len);
  int Close();
};

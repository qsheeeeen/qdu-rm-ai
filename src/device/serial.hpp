#pragma once

typedef enum {
  kBR9600,
  KBR115200,
} BaudRate;

class Serial {
 private:
  int dev_;

 public:
  Serial(int index);
  ~Serial();
  bool Config(bool parity, bool stop_bit, bool flow_ctrl, BaudRate br);
  void Trans(char buff[], int len);
  void Recv(char buff[], int len);
};

#pragma once

class Robot {
 private:
  float place_holder;

 public:
  Robot();
  ~Robot();

  bool Connect();
  bool Disconnect();

  bool Parse();
  bool Send();
};

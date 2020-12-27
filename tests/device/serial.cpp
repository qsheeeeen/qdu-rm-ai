#include "serial.hpp"

#include "gtest/gtest.h"

const std::string msg = "hello\n";

TEST(TestSerial, TestTrans) {
  Serial com("/dev/stdout");
  ASSERT_EQ(com.Trans(msg.c_str(), msg.length()), msg.length())
      << "Can not transmit message.";
}

TEST(TestSerial, TestConfig) {
  Serial com("/dev/ttyS0");
  ASSERT_TRUE(com.Config(true, true, true, BaudRate::kBR9600))
      << "Can not config serial port.";
}
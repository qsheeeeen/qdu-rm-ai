#include "serial.hpp"

#include "gtest/gtest.h"

const std::string msg = "hello\n";

TEST(TestSerial, TestTrans) {
  Serial com("/dev/stdout");
  ASSERT_EQ(com.Trans(msg.c_str(), msg.length()), msg.length())
      << "Can not transmit message.";
}

TEST(TestSerial, TestConfig) {
  /* Serial _/dev/ttyTHS2_ is occupied by tests/robot.cpp */
  /* Here should be _/dev/ttyTHS2_ */
  Serial com("/dev/ttyS0");
  ASSERT_TRUE(com.Config(true, StopBits::kSTOP_BITS_1, DataLength::kDATA_LEN_7,
                         true, BaudRate::kBAUD_RATE_460800))
      << "Can not config serial port.";
}
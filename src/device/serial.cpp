#include "serial.hpp"

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

#include "spdlog/spdlog.h"

/**
 * @brief Construct a new Serial object
 *
 */
Serial::Serial() {
  dev_ = -1;
  SPDLOG_TRACE("Constructed.");
}

/**
 * @brief Construct a new Serial object
 *
 * @param dev_path 具体要读写的串口设备
 */
Serial::Serial(const std::string& dev_path) {
  dev_ = open(dev_path.c_str(), O_RDWR);

  if (dev_ < 0)
    SPDLOG_ERROR("Can't open Serial device.");
  else
    Config();

  SPDLOG_TRACE("Constructed.");
}

/**
 * @brief Destroy the Serial object
 *
 */
Serial::~Serial() {
  Close();
  SPDLOG_TRACE("Destructed.");
}

/**
 * @brief 打开串口
 *
 * @param dev_path 具体要读写的串口设备
 */
void Serial::Open(const std::string& dev_path) {
  dev_ = open(dev_path.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);

  if (dev_ < 0) SPDLOG_ERROR("Can't open Serial device.");
}

/**
 * @brief 检查串口是否打开
 *
 * @return true 已打开
 * @return false 未打开
 */
bool Serial::IsOpen() { return (dev_ > 0); }

/**
 * @brief 配置串口
 *
 * @param parity
 * @param stop_bit 停止位
 * @param flow_ctrl 流控制
 * @param baud_rate 波特率
 * @return true 配置成功
 * @return false 配置失败
 */
bool Serial::Config(bool parity, StopBits stop_bit, DataLength data_length,
                    bool flow_ctrl, BaudRate baud_rate) {
  struct termios tty_cfg;

  SPDLOG_INFO(
      "parity={}, stop_bit={}, data_length={}, flow_ctrl={}, "
      "baud_rate={}",
      parity, stop_bit, data_length, flow_ctrl, baud_rate);

  if (tcgetattr(dev_, &tty_cfg)) {
    SPDLOG_ERROR("Error {} from tcgetattr: {}.", errno, std::strerror(errno));
    return false;
  }

  if (parity)
    tty_cfg.c_cflag |= PARENB;
  else {
    tty_cfg.c_cflag &= ~PARENB;
    tty_cfg.c_iflag &= ~INPCK;
  }
  switch (stop_bit) {
    case StopBits::kSTOP_BITS_1:
      tty_cfg.c_cflag &= ~CSTOPB;
      break;
    case StopBits::kSTOP_BITS_2:
      tty_cfg.c_cflag |= CSTOPB;
      break;
  }

  if (flow_ctrl)
    tty_cfg.c_cflag |= CRTSCTS;
  else
    tty_cfg.c_cflag &= ~CRTSCTS;

  switch (baud_rate) {
    case BaudRate::kBAUD_RATE_9600:
      cfsetispeed(&tty_cfg, B9600);
      cfsetospeed(&tty_cfg, B9600);
      break;
    case BaudRate::kBAUD_RATE_115200:
      cfsetispeed(&tty_cfg, B115200);
      cfsetospeed(&tty_cfg, B115200);
      break;
    case BaudRate::kBAUD_RATE_460800:
      cfsetispeed(&tty_cfg, B460800);
      cfsetospeed(&tty_cfg, B460800);
      break;
  }
  // 一般必设置的标志
  tty_cfg.c_cflag |= (CLOCAL | CREAD);
  tty_cfg.c_oflag &= ~(OPOST);
  tty_cfg.c_lflag &= ~(ECHO | ICANON | IEXTEN | ISIG);
  tty_cfg.c_iflag &= ~(ICRNL | INLCR | IGNCR | IXON | IXOFF | IXANY);
  
  // 清空输入输出缓冲区
  tcflush(dev_, TCIOFLUSH);
  
  switch (data_length) {
    case DataLength::kDATA_LEN_5:
      tty_cfg.c_cflag |= CS5;
      break;
    case DataLength::kDATA_LEN_6:
      tty_cfg.c_cflag |= CS6;
      break;
    case DataLength::kDATA_LEN_7:
      tty_cfg.c_cflag |= CS7;
      break;
    case DataLength::kDATA_LEN_8:
      tty_cfg.c_cflag &= ~CSIZE;
      tty_cfg.c_cflag |= CS8;
      break;
  }

  if (tcsetattr(dev_, TCSANOW, &tty_cfg) != 0) {
    SPDLOG_ERROR("Error {d} from tcsetattr: {s}\n", errno,
                 std::strerror(errno));
    return false;
  }
  return true;
}

/**
 * @brief 发送
 *
 * @param buff 缓冲区地址
 * @param len 缓冲区长度
 * @return ssize_t 已发送的长度
 */
std::size_t Serial::Trans(const void* buff, std::size_t len) {
  std::lock_guard<std::mutex> lock(mutex_);
  return write(dev_, buff, len);
}

/**
 * @brief 接收
 *
 * @param buff 缓冲区地址
 * @param len 缓冲区长度
 * @return ssize_t 已发送的长度
 */
std::size_t Serial::Recv(void* buff, std::size_t len) {
  std::lock_guard<std::mutex> lock(mutex_);
  return read(dev_, buff, len);
}

/**
 * @brief 关闭
 *
 * @return int 状态代码
 */
int Serial::Close() { return close(dev_); }

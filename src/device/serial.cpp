#include "serial.hpp"

#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <termios.h>
#include <unistd.h>

#include "spdlog/spdlog.h"

/**
 * @brief Construct a new Serial object
 *
 */
Serial::Serial() {
  SPDLOG_DEBUG("[Serial] Constructing.");
  dev_ = -1;
  SPDLOG_DEBUG("[Serial] Constructed.");
}

/**
 * @brief Construct a new Serial object
 *
 * @param dev_path 具体要读写的串口设备
 */
Serial::Serial(const std::string& dev_path) {
  SPDLOG_DEBUG("[Serial] Constructing.");

  dev_ = open(dev_path.c_str(), O_RDWR);

  if (dev_ < 0)
    SPDLOG_ERROR("[Serial] Can't open Serial device.");
  else
    Config(false, false, false, BaudRate::KBR115200);

  SPDLOG_DEBUG("[Serial] Constructed.");
}

/**
 * @brief Destroy the Serial object
 *
 */
Serial::~Serial() {
  SPDLOG_DEBUG("[Serial] Destructing.");
  Close();
  SPDLOG_DEBUG("[Serial] Destructed.");
}

/**
 * @brief 打开串口
 *
 * @param dev_path 具体要读写的串口设备
 */
void Serial::Open(const std::string& dev_path) {
  dev_ = open(dev_path.c_str(), O_RDWR);

  if (dev_ < 0) SPDLOG_ERROR("[Serial] Can't open Serial device.");
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
 * @param br 波特率
 * @return true 配置成功
 * @return false 配置失败
 */
bool Serial::Config(bool parity, bool stop_bit, bool flow_ctrl, BaudRate br) {
  struct termios tty_cfg;

  SPDLOG_INFO("[Serial] parity={}, stop_bit={}, flow_ctrl={}, br={}", parity,
              stop_bit, flow_ctrl, br);

  if (tcgetattr(dev_, &tty_cfg)) {
    SPDLOG_ERROR("[Serial] Error {} from tcgetattr: {}.", errno,
                 strerror(errno));
    return false;
  }

  if (parity)
    tty_cfg.c_cflag |= PARENB;
  else
    tty_cfg.c_cflag &= ~PARENB;

  if (stop_bit)
    tty_cfg.c_cflag |= CSTOPB;
  else
    tty_cfg.c_cflag &= ~CSTOPB;

  tty_cfg.c_cflag |= CS8;

  if (flow_ctrl)
    tty_cfg.c_cflag |= CRTSCTS;
  else
    tty_cfg.c_cflag &= ~CRTSCTS;

  switch (br) {
    case BaudRate::kBR9600:
      cfsetispeed(&tty_cfg, B9600);
      cfsetospeed(&tty_cfg, B9600);
      break;
    case BaudRate::KBR115200:
      cfsetispeed(&tty_cfg, B115200);
      cfsetospeed(&tty_cfg, B115200);
      break;
  }

  if (tcsetattr(dev_, TCSANOW, &tty_cfg) != 0) {
    SPDLOG_ERROR("[Serial] Error {d} from tcsetattr: {s}\n", errno,
                 strerror(errno));
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
ssize_t Serial::Trans(const void* buff, size_t len) {
  return write(dev_, buff, len);
}

/**
 * @brief 接收
 *
 * @param buff 缓冲区地址
 * @param len 缓冲区长度
 * @return ssize_t 已发送的长度
 */
ssize_t Serial::Recv(void* buff, size_t len) { return read(dev_, buff, len); }

/**
 * @brief 关闭
 *
 * @return int 状态代码
 */
int Serial::Close() { return close(dev_); }

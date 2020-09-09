#include "serial.hpp"

#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <termios.h>
#include <unistd.h>

#include "spdlog/spdlog.h"

Serial::Serial() {
  SPDLOG_DEBUG("[Serial] Constructing.");
  dev_ = -1;
  SPDLOG_DEBUG("[Serial] Constructed.");
}

Serial::Serial(const std::string& dev_path) {
  SPDLOG_DEBUG("[Serial] Constructing.");

  dev_ = open(dev_path.c_str(), O_RDWR);

  if (dev_ < 0)
    SPDLOG_ERROR("[Serial] Can't open Serial device.");
  else
    Config(false, false, false, BaudRate::KBR115200);

  SPDLOG_DEBUG("[Serial] Constructed.");
}

Serial::~Serial() {
  SPDLOG_DEBUG("[Serial] Destructing.");
  close(dev_);
  SPDLOG_DEBUG("[Serial] Destructed.");
}

void Serial::Open(const std::string& dev_path) {
  dev_ = open(dev_path.c_str(), O_RDWR);

  if (dev_ < 0) SPDLOG_ERROR("[Serial] Can't open Serial device.");
}

bool Serial::IsOpen() { return (dev_ > 0); }

bool Serial::Config(bool parity, bool stop_bit, bool flow_ctrl, BaudRate br) {
  struct termios tty_cfg;

  SPDLOG_DEBUG("[Serial] parity={}, stop_bit={}, flow_ctrl={}, br={}", parity,
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

ssize_t Serial::Trans(const void* buff, size_t len) {
  return write(dev_, buff, len);
}

ssize_t Serial::Recv(void* buff, size_t len) { return read(dev_, buff, len); }

int Serial::Close() { return close(dev_); }

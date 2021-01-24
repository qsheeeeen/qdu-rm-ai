#include "mcu.hpp"

#include "spdlog/spdlog.h"

void MCU::ComThread() {
  SPDLOG_DEBUG("[ComThread] Started.");

  while (continue_parse_) {
    uint16_t ID;
    serial_.Recv((uint16_t *)&ID, sizeof(uint16_t));

    if (AI_ID_REF == ID) {
      status_refe_.id = ID;
      serial_.Recv((char *)&status_refe_.data, sizeof(status_refe_.data));
      serial_.Recv((char *)&status_refe_.crc16, sizeof(status_refe_.crc16));

      serial_.Recv((char *)&status_mcu_, sizeof(status_mcu_));

      if (CRC16_Verify((const uint8_t *)&status_refe_, sizeof(status_refe_))) {
        Protocol_AI_t result;
        /**
         * 运算部分
         *
         */
        commandq_.push(result);
      }
    } else if (AI_ID_MCU == ID) {
      status_mcu_.id = ID;
      serial_.Recv((char *)&status_mcu_.data, sizeof(status_mcu_.data));
      serial_.Recv((char *)&status_mcu_.crc16, sizeof(status_mcu_.crc16));

      if (CRC16_Verify((const uint8_t *)&status_mcu_, sizeof(status_mcu_))) {
        Protocol_AI_t result;
        /**
         * 运算部分
         *
         */
        commandq_.push(result);
      }
    }
    commandq_mutex_.lock();
    if (!commandq_.empty()) {
      serial_.Trans((char *)&commandq_.front(), sizeof(Protocol_AI_t));
      commandq_.pop();
    }
    commandq_mutex_.unlock();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  SPDLOG_DEBUG("[ComThread] Stoped.");
}

void MCU::CommandThread() {
  SPDLOG_DEBUG("[CommandThread] Started.");

  SPDLOG_DEBUG("[CommandThread] Stoped.");
}

MCU::MCU(const std::string &dev_path) {
  serial_.Open(dev_path);
  serial_.Config();
  if (!serial_.IsOpen()) {
    SPDLOG_ERROR("Can't open device.");
  }

  continue_parse_ = true;
  parse_thread_ = std::thread(&MCU::ComThread, this);

  SPDLOG_TRACE("Constructed.");
}

MCU::~MCU() {
  serial_.Close();

  continue_parse_ = false;
  parse_thread_.join();
  SPDLOG_TRACE("Destructed.");
}

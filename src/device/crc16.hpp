#pragma once

#include <cstdbool>
#include <cstdint>

namespace crc16 {

uint16_t CRC16_Calc(const uint8_t *buf, std::size_t len, uint16_t crc);
bool CRC16_Verify(const uint8_t *buf, std::size_t len);

}

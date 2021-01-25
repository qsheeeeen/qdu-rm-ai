#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <cstdbool>
#include <cstdint>

uint16_t CRC16_Calc(const uint8_t *buf, std::size_t len, uint16_t crc);
bool CRC16_Verify(const uint8_t *buf, std::size_t len);

#ifdef __cplusplus
}
#endif

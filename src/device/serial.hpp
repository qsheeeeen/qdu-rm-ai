#pragma once

#include <mutex>
#include <string>

/* 波特率 */
enum class BaudRate {
  kBAUD_RATE_9600,
  kBAUD_RATE_115200,
  kBAUD_RATE_460800
};

/* 停止位数量 */
enum class StopBits {
  kSTOP_BITS_1,
  kSTOP_BITS_2,
};

/* 有效数据 */
enum class DataLength {
  kDATA_LEN_5,
  kDATA_LEN_6,
  kDATA_LEN_7,
  kDATA_LEN_8,
};

/* 串口 */
class Serial {
 private:
  int dev_;
  std::mutex mutex_;

 public:
  /**
   * @brief Construct a new Serial object
   *
   */
  Serial();

  /**
   * @brief Construct a new Serial object
   *
   * @param dev_path 具体要读写的串口设备
   */
  Serial(const std::string& dev_path);

  /**
   * @brief Destroy the Serial object
   *
   */
  ~Serial();

  /**
   * @brief 打开串口
   *
   * @param dev_path 具体要读写的串口设备
   */
  void Open(const std::string& dev_path);

  /**
   * @brief 检查串口是否打开
   *
   * @return true 已打开
   * @return false 未打开
   */
  bool IsOpen();

  /**
   * @brief 配置串口
   *
   * @param parity 校验
   * @param stop_bit 停止位
   * @param flow_ctrl 流控制
   * @param br 波特率
   * @return true 配置成功
   * @return false 配置失败
   */
  bool Config(bool parity = false, StopBits stop_bit = StopBits::kSTOP_BITS_1,
              DataLength data_length = DataLength::kDATA_LEN_8,
              bool flow_ctrl = false,
              BaudRate baud_rate = BaudRate::kBAUD_RATE_460800);

  /**
   * @brief 发送
   *
   * @param buff 缓冲区地址
   * @param len 缓冲区长度
   * @return std::size_t 已发送的长度
   */
  std::size_t Trans(const void* buff, std::size_t len);

  /**
   * @brief 接收
   *
   * @param buff 缓冲区地址
   * @param len 缓冲区长度
   * @return std::size_t 已发送的长度
   */
  std::size_t Recv(void* buff, std::size_t len);

  /**
   * @brief 关闭
   *
   * @return int 状态代码
   */
  int Close();
};

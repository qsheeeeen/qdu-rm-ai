#pragma once

#include <string>

/**
 * @brief 波特率
 *
 */
enum class BaudRate {
  kBR9600,
  kBR115200,
};

/**
 * @brief 串口
 *
 */
class Serial {
 private:
  int dev_;

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
  bool Config(bool parity = false, bool stop_bit = false,
              bool flow_ctrl = false, BaudRate br = BaudRate::kBR115200);

  /**
   * @brief 发送
   *
   * @param buff 缓冲区地址
   * @param len 缓冲区长度
   * @return ssize_t 已发送的长度
   */
  ssize_t Trans(const void* buff, size_t len);

  /**
   * @brief 接收
   *
   * @param buff 缓冲区地址
   * @param len 缓冲区长度
   * @return ssize_t 已发送的长度
   */
  ssize_t Recv(void* buff, size_t len);

  /**
   * @brief 关闭
   *
   * @return int 状态代码
   */
  int Close();
};

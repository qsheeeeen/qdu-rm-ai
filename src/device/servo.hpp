#pragma once

#include <mutex>
#include <string>

/* 舵机 */
class Servo {
 private:
  int dev_;
  std::mutex mutex_;

 public:
  /**
   * @brief Construct a new Servo object
   *
   */
  Servo();

  /**
   * @brief Construct a new Servo object
   *
   * @param dev_path 具体要读写的舵机设备
   */
  Servo(const std::string& dev_path);

  /**
   * @brief Destroy the Servo object
   *
   */
  ~Servo();

  /**
   * @brief 打开舵机
   *
   * @param dev_path 具体要读写的舵机设备
   */
  void Open(const std::string& dev_path);

  /**
   * @brief 检查舵机是否打开
   *
   * @return true 已打开
   * @return false 未打开
   */
  bool IsOpen();

  /**
   * @brief 配置舵机
   *
   * @param hi_width 最大角度时的占空时间
   * @param lo_width 最小角度时的占空时间
   * @param hi_angle 最大角度
   * @param lo_angle 最小角度
   * @return true 成功
   * @return false 是失败
   */
  bool Config(float hi_width, float lo_width, float hi_angle, float lo_angle);

  /**
   * @brief
   *
   * @param angle 转到的角度
   * @return true 成功
   * @return false 是失败
   */
  bool Set(float angle);

  /**
   * @brief 关闭
   *
   * @return int 状态代码
   */
  int Close();
};

#pragma once

#include <deque>
#include <mutex>
#include <thread>

#include "camera.hpp"
#include "opencv2/core/mat.hpp"

class RaspiCamera : public Camera {
 private:
  /**
   * @brief 相机初始化前的准备工作
   *
   */
  void Prepare();

 public:
  /**
   * @brief Construct a new RaspiCamera object
   *
   */
  RaspiCamera();

  /**
   * @brief Construct a new RaspiCamera object
   *
   * @param index 相机索引号
   * @param height 输出图像高度
   * @param width 输出图像宽度
   */
  RaspiCamera(unsigned int index, unsigned int height, unsigned int width);

  /**
   * @brief Destroy the RaspiCamera object
   *
   */
  ~RaspiCamera();

  /**
   * @brief 设置相机参数
   *
   * @param height 输出图像高度
   * @param width 输出图像宽度
   */
  void Setup(unsigned int height, unsigned int width);

  /**
   * @brief 打开相机设备
   *
   * @param index 相机索引号
   * @return int 状态代码
   */
  int Open(unsigned int index);

  /**
   * @brief Get the Frame object
   *
   * @return cv::Mat 拍摄的图像
   */
  cv::Mat GetFrame();

  /**
   * @brief 关闭相机设备
   *
   * @return int 状态代码
   */
  int Close();

  /**
   * @brief 相机标定
   *
   */
  void Calibrate();
};

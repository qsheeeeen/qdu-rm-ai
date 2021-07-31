#pragma once

#include <deque>
#include <mutex>
#include <thread>

#include "MvCameraControl.h"
#include "camera.hpp"
#include "opencv2/core/mat.hpp"

class HikCamera : public Camera {
 private:
  MV_CC_DEVICE_INFO_LIST mv_dev_list_;
  void *camera_handle_ = nullptr;
  MV_FRAME_OUT raw_frame;

  void GrabPrepare();
  void GrabLoop();

  /**
   * @brief 相机初始化前的准备工作
   *
   */
  void Prepare();

 public:
  /**
   * @brief Construct a new HikCamera object
   *
   */
  HikCamera();

  /**
   * @brief Construct a new HikCamera object
   *
   * @param index 相机索引号
   * @param height 输出图像高度
   * @param width 输出图像宽度
   */
  HikCamera(unsigned int index, unsigned int height, unsigned int width);

  /**
   * @brief Destroy the HikCamera object
   *
   */
  ~HikCamera();

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
  bool Open(unsigned int index);

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

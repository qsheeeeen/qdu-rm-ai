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
  bool OpenPrepare(unsigned int index);

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
   * @brief 关闭相机设备
   *
   * @return int 状态代码
   */
  int Close();
};

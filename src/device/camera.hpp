#pragma once

#include <deque>
#include <mutex>
#include <thread>

#include "opencv2/core/mat.hpp"
#include "spdlog/spdlog.h"

class Camera {
 public:
  unsigned int frame_h_, frame_w_;

  bool grabing = false;
  std::thread grab_thread_;
  std::mutex frame_stack_mutex_;
  std::deque<cv::Mat> frame_stack_;

  virtual void GrabPrepare() = 0;
  virtual void GrabLoop() = 0;

  void GrabThread() {
    SPDLOG_DEBUG("[GrabThread] Started.");
    GrabPrepare();
    while (grabing) GrabLoop();

    SPDLOG_DEBUG("[GrabThread] Stoped.");
  }

  /**
   * @brief 设置相机参数
   *
   * @param height 输出图像高度
   * @param width 输出图像宽度
   */
  virtual void Setup(unsigned int height, unsigned int width) = 0;

  /**
   * @brief 打开相机设备
   *
   * @param index 相机索引号
   * @return true 打开成功
   * @return false 打开失败
   */
  virtual bool Open(unsigned int index) = 0;

  /**
   * @brief Get the Frame object
   *
   * @return cv::Mat 拍摄的图像
   */
  virtual cv::Mat GetFrame() = 0;

  /**
   * @brief 关闭相机设备
   *
   * @return int 状态代码
   */
  virtual int Close() = 0;
};

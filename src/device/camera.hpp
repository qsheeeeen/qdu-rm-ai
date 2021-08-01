#pragma once

#include <deque>
#include <mutex>
#include <thread>

#include "opencv2/core/mat.hpp"
#include "spdlog/spdlog.h"

class Camera {
 private:
  virtual void GrabPrepare() = 0;
  virtual void GrabLoop() = 0;

  void GrabThread() {
    SPDLOG_DEBUG("[GrabThread] Started.");
    GrabPrepare();
    while (grabing) GrabLoop();

    SPDLOG_DEBUG("[GrabThread] Stoped.");
  }

  virtual bool OpenPrepare(unsigned int index) = 0;

 public:
  unsigned int frame_h_, frame_w_;

  bool grabing = false;
  std::thread grab_thread_;
  std::mutex frame_stack_mutex_;
  std::deque<cv::Mat> frame_stack_;

  /**
   * @brief 设置相机参数
   *
   * @param height 输出图像高度
   * @param width 输出图像宽度
   */
  void Setup(unsigned int height, unsigned int width) {
    frame_h_ = height;
    frame_w_ = width;

    // TODO: 配置相机输入输出
  }

  /**
   * @brief 打开相机设备
   *
   * @param index 相机索引号
   * @return true 打开成功
   * @return false 打开失败
   */
  bool Open(unsigned int index) {
    if (OpenPrepare(index)) {
      grabing = true;
      grab_thread_ = std::thread(&Camera::GrabThread, this);
      return true;
    }
    return false;
  }

  /**
   * @brief Get the Frame object
   *
   * @return cv::Mat 拍摄的图像
   */
  virtual cv::Mat GetFrame() {
    cv::Mat frame;

    std::lock_guard<std::mutex> lock(frame_stack_mutex_);
    if (!frame_stack_.empty()) {
      frame = frame_stack_.front();
      frame_stack_.clear();
    } else {
      SPDLOG_ERROR("Empty frame stack!");
    }
    return frame;
  }
  /**
   * @brief 关闭相机设备
   *
   * @return int 状态代码
   */
  virtual int Close() = 0;
};

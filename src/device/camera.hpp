#pragma once

#include <stack>
#include <thread>

#include "MvCameraControl.h"
#include "opencv2/core/mat.hpp"

class Camera {
 private:
  MV_CC_DEVICE_INFO_LIST mv_dev_list_;
  void *camera_handle_ = nullptr;
  MV_FRAME_OUT raw_frame;

  unsigned int frame_h_, frame_w_;

  bool grabing = false;
  std::thread grab_thread_;
  std::stack<cv::Mat> frame_stack_;

  void GrabThread();
  void Prepare();

 public:
  Camera();
  Camera(unsigned int index, unsigned int height, unsigned int width);
  ~Camera();

  void Setup(unsigned int height, unsigned int width);
  int Open(unsigned int index);

  cv::Mat GetFrame();
  int Close();
};

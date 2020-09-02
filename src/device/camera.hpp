#pragma once

#include <thread>

#include "MvCameraControl.h"
#include "opencv2/opencv.hpp"

class Camera {
 private:
  MV_CC_DEVICE_INFO_LIST mv_dev_list_;
  void *camera_handle_ = nullptr;

  bool continue_capture_ = false;
  std::thread capture_thread_;

  unsigned int out_h_, out_w_;

  cv::Mat image;

  void WorkThread();
  void PrintDeviceInfo(MV_CC_DEVICE_INFO *mv_dev_info);
  void Prepare();

 public:
  Camera(unsigned int out_h, unsigned int out_w);
  Camera(unsigned int index, unsigned int out_h, unsigned int out_w);
  ~Camera();

  void Open(unsigned int index);
  bool GetFrame(void *output);
  int Close();
};

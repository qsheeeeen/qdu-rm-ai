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

  cv::Mat image(608, 608, CV_32FC3);

  void WorkThread();
  void PrintDeviceInfo(MV_CC_DEVICE_INFO *mv_dev_info);
  void Prepare();

 public:
  Camera();
  Camera(unsigned int index);
  ~Camera();

  void Open(unsigned int index);
  bool GetFrame(void *output);
  int Close();
};

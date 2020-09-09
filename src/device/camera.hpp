#pragma once

#include "MvCameraControl.h"
#include "opencv2/core/mat.hpp"

class Camera {
 private:
  MV_CC_DEVICE_INFO_LIST mv_dev_list_;
  void *camera_handle_ = nullptr;

  unsigned int out_h_, out_w_;

  void Prepare();

 public:
  Camera();
  Camera(unsigned int out_h, unsigned int out_w);
  Camera(unsigned int index, unsigned int out_h, unsigned int out_w);
  ~Camera();

  void Setup(unsigned int out_h, unsigned int out_w);
  int Open(unsigned int index);

  cv::Mat GetFrame();
  int Close();
};

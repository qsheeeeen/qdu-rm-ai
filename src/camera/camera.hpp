#pragma once

#include "MvCameraControl.h"
#include "opencv2/opencv.hpp"

class Camera {
 private:
  //  OpenCV
  cv::VideoCapture video_cap;
  cv::Mat frame_in;
  cv::Mat hist_in;
  cv::Mat hist_target;

  const int hist_ch[3] = {0, 1, 2};
  const int hist_size[3] = {256, 256, 256};
  float hist_ch_range[2] = {0., 255.};
  const float *hist_range[3] = {hist_ch_range, hist_ch_range, hist_ch_range};

  float gamma = 1.;
  cv::Mat lut;

  //  MVC SDK
  unsigned int payload_size = 0;
  MV_CC_DEVICE_INFO *mv_dev_info;
  MV_CC_DEVICE_INFO_LIST mv_dev_list;
  MVCC_INTVALUE init_val;
  void *camera_handle = nullptr;

  void PrintDeviceInfo();

  void CreateLUT();
  void AppyLUT();

  void LoadTargetHist();

  void GetFrame(cv::Mat &output);

  void MatchTargetHist();

 public:
  Camera(unsigned int index);
  ~Camera();
  void *GetCameraHandle();
  bool Capture();
  void Preprocess();
  void CalcTargetHist();
};

#pragma once

#include "opencv2/opencv.hpp"

class LightBar {
 private:
  cv::RotatedRect rect_;

 public:
  LightBar();
  LightBar(const cv::RotatedRect &rect);
  ~LightBar();

  void Init(const cv::RotatedRect &rect);

  const cv::Point2f &Center();
  float Angle();
  float Length();
};

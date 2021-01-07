#pragma once

#include "opencv2/opencv.hpp"

class LighjtBar {
 private:
  cv::RotatedRect rect_;
  cv::Point2f center_;

 public:
  LighjtBar();
  LighjtBar(const cv::RotatedRect &rect);
  ~LighjtBar();

  void Init(const cv::RotatedRect &rect);

  const cv::Point2f &Center();
  float Angle();
  float Length();
};

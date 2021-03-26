#pragma once

#include <vector>

#include "opencv2/opencv.hpp"

class LightBar {
 private:
  cv::RotatedRect rect_;

 public:
  LightBar();
  LightBar(const cv::RotatedRect &rect);
  ~LightBar();

  void Init(const cv::RotatedRect &rect);

  const cv::Point2f &Center() const;
  std::vector<cv::Point2f> Vertices() const;
  double Angle() const;
  double Area() const;
  double AspectRatio() const;
  double Length() const;
};

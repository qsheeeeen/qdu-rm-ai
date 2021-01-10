#include "light_bar.hpp"

#include "opencv2/opencv.hpp"
#include "spdlog/spdlog.h"

LightBar::LightBar() { SPDLOG_DEBUG("Constructed."); }

LightBar::LightBar(const cv::RotatedRect &rect) {
  Init(rect);
  SPDLOG_DEBUG("Constructed.");
}

LightBar::~LightBar() { SPDLOG_DEBUG("Destructed."); }

void LightBar::Init(const cv::RotatedRect &rect) {
  rect_ = rect;
  if (rect_.angle > 90.) {
    rect_.angle -= 180.;
    std::swap(rect_.size.height, rect_.size.width);
  }
  SPDLOG_DEBUG("Inited.");
}

const cv::Point2f &LightBar::Center() { return rect_.center; }

std::vector<cv::Point2f> LightBar::Vertices() {
  std::vector<cv::Point2f> vertices(4);
  rect_.points(vertices.data());
  return vertices;
}

float LightBar::Angle() {
  SPDLOG_DEBUG("rect_.angle: {}", rect_.angle);
  return rect_.angle;
}

float LightBar::Area() { return rect_.size.area(); }

float LightBar::AspectRatio() {
  float aspect_ratio = rect_.size.aspectRatio();
  SPDLOG_DEBUG("aspect_ratio: {}", aspect_ratio);
  return aspect_ratio;
}

float LightBar::Length() {
  SPDLOG_DEBUG("rect_.size (h,w): ({}, {})", rect_.size.height,
               rect_.size.width);
  return std::max(rect_.size.height, rect_.size.width);
}

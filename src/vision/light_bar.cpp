#include "light_bar.hpp"

#include "opencv2/opencv.hpp"
#include "spdlog/spdlog.h"

LightBar::LightBar() { SPDLOG_TRACE("Constructed."); }

LightBar::LightBar(const cv::RotatedRect &rect) {
  Init(rect);
  SPDLOG_TRACE("Constructed.");
}

LightBar::~LightBar() { SPDLOG_TRACE("Destructed."); }

void LightBar::Init(const cv::RotatedRect &rect) {
  rect_ = rect;
  if (rect_.size.width > rect_.size.height) {
    rect_.angle -= 90.;
    std::swap(rect_.size.width, rect_.size.height);
  }
  SPDLOG_DEBUG("Inited.");
}

const cv::Point2f &LightBar::Center() const { return rect_.center; }

std::vector<cv::Point2f> LightBar::Vertices() const {
  std::vector<cv::Point2f> vertices(4);
  rect_.points(vertices.data());
  return vertices;
}

double LightBar::Angle() const {
  if (rect_.angle > 90.) {
    return rect_.angle - 180.;
  } else if (rect_.angle > 270.) {
    return 360. - rect_.angle;
  } else {
    return rect_.angle;
  }
}

double LightBar::Area() const { return rect_.size.area(); }

double LightBar::AspectRatio() const {
  double aspect_ratio = std::max(rect_.size.height, rect_.size.width) /
                        std::min(rect_.size.height, rect_.size.width);
  SPDLOG_DEBUG("aspect_ratio: {}", aspect_ratio);
  return aspect_ratio;
}

double LightBar::Length() const {
  SPDLOG_DEBUG("rect_.size (h,w): ({}, {})", rect_.size.height,
               rect_.size.width);
  return std::max(rect_.size.height, rect_.size.width);
}

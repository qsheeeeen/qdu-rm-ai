#include "light_bar.hpp"

#include "opencv2/opencv.hpp"
#include "spdlog/spdlog.h"

LightBar::LightBar() { SPDLOG_DEBUG("[LightBar] Constructed."); }

LightBar::LightBar(const cv::RotatedRect &rect) : rect_(rect) {
  SPDLOG_DEBUG("[LightBar] Constructed.");
}

LightBar::~LightBar() { SPDLOG_DEBUG("[LightBar] Destructed."); }

void LightBar::Init(const cv::RotatedRect &rect) {
  rect_ = rect;
  SPDLOG_DEBUG("[LightBar] Inited.");
}

const cv::Point2f &LightBar::Center() { return rect_.center; }

float LightBar::Angle() {
  SPDLOG_DEBUG("[LightBar] rect_.angle: {}", rect_.angle);

  if (rect_.angle > 90.f)
    return rect_.angle - 180.f;
  else
    return rect_.angle;
}

float LightBar::Length() {
  SPDLOG_DEBUG("[LightBar] rect_.size (h,w): ({}, {})", rect_.size.height,
               rect_.size.width);
  return std::max(rect_.size.height, rect_.size.width);
}
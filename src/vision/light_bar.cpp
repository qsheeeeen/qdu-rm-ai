#include "light_bar.hpp"

#include "opencv2/opencv.hpp"
#include "spdlog/spdlog.h"

LighjtBar::LighjtBar() { SPDLOG_DEBUG("[LighjtBar] Constructed."); }

LighjtBar::LighjtBar(const cv::RotatedRect &rect) : rect_(rect) {
  SPDLOG_DEBUG("[LighjtBar] Constructed.");
}

LighjtBar::~LighjtBar() { SPDLOG_DEBUG("[LighjtBar] Destructed."); }

void LighjtBar::Init(const cv::RotatedRect &rect) {
  rect_ = rect;
  SPDLOG_DEBUG("[LighjtBar] Inited.");
}

const cv::Point2f &LighjtBar::Center() { return rect_.center; }

float LighjtBar::Angle() {
  SPDLOG_DEBUG("[LighjtBar] rect_.angle: {}", rect_.angle);

  if (rect_.angle > 90.f)
    return rect_.angle - 180.f;
  else
    return rect_.angle;
}

float LighjtBar::Length() {
  SPDLOG_DEBUG("[LighjtBar] rect_.size (h,w): ({}, {})", rect_.size.height,
               rect_.size.width);
  return std::max(rect_.size.height, rect_.size.width);
}
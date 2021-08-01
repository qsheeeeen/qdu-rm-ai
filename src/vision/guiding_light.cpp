#include "guiding_light.hpp"

#include "opencv2/opencv.hpp"
#include "spdlog/spdlog.h"

GuidingLight::GuidingLight() { SPDLOG_TRACE("Constructed."); }

GuidingLight::GuidingLight(const cv::Point2f &center, float radius) {
  (void)center;
  (void)radius;
}

GuidingLight::~GuidingLight() { SPDLOG_TRACE("Destructed."); }

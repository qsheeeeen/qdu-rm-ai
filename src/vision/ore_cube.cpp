#include "ore_cube.hpp"

#include "opencv2/opencv.hpp"
#include "spdlog/spdlog.h"

OreCube::OreCube() { SPDLOG_TRACE("Constructed."); }

OreCube::OreCube(const cv::Point2f &center, float radius) {
  (void)center;
  (void)radius;
}

OreCube::~OreCube() { SPDLOG_TRACE("Destructed."); }

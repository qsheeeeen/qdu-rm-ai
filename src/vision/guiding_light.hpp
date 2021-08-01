#pragma once

#include <vector>

#include "common.hpp"
#include "light_bar.hpp"
#include "object.hpp"
#include "opencv2/opencv.hpp"

class GuidingLight : public ImageObject, public PhysicObject {
 public:
  GuidingLight();
  GuidingLight(const cv::Point2f &center, float radius);
  ~GuidingLight();
};

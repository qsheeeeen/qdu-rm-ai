#pragma once

#include "object.hpp"
#include "opencv2/opencv.hpp"

class OreCube : public ImageObject, public PhysicObject {
 public:
  OreCube();
  OreCube(const cv::Point2f &center, float radius);
  ~OreCube();
};

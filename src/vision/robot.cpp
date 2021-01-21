#include "robot.hpp"

#include "opencv2/opencv.hpp"
#include "spdlog/spdlog.h"

Robot::Robot() { SPDLOG_TRACE("Constructed."); }

Robot::Robot(Armor armor) {
  Init(armor);
  SPDLOG_TRACE("Constructed.");
}

Robot::Robot(std::vector<Armor> armors) {
  Init(armors);
  SPDLOG_TRACE("Constructed.");
}

Robot::~Robot() { SPDLOG_TRACE("Destructed."); }

void Robot::Init(Armor armor) {
  armors_.emplace_back(armor);
  SPDLOG_DEBUG("Inited.");
}

void Robot::Init(std::vector<Armor> armors) {
  armors_.insert(armors_.end(), armors.begin(), armors.end());
  SPDLOG_DEBUG("Inited.");
}

game::Team Robot::Team() { return armors_.front().Team(); }
game::Model Robot::Model() { return armors_.front().Model(); }
cv::Point3f Robot::Center() {}
std::vector<cv::Point2f> Robot::Vertices() {}
cv::Mat Robot::Rotation() {}
cv::Vec3d Robot::RotationAxis() {}
cv::Mat Robot::Translation() {}
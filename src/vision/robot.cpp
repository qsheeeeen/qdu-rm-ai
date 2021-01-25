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

game::Team Robot::GetTeam() { return armors_.front().GetTeam(); }
game::Model Robot::GetModel() { return armors_.front().GetModel(); }
cv::Point3f Robot::Center3D() {}

std::vector<cv::Point3f> Robot::Vertices3D() {
  cv::Mat point_mat = cv::Mat(armors_.front().Vertices3D()).reshape(1).t();
  cv::Point3f word(cv::Mat(point_mat * armors_.front().GetRotMat() +
                           armors_.front().GetTransVec()));
}

cv::Mat Robot::GetRotMat() {}
cv::Vec3d Robot::RotationAxis() {}
cv::Mat Robot::Translation() {}
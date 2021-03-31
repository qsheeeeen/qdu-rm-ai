#include "buff.hpp"

#include "opencv2/opencv.hpp"
#include "spdlog/spdlog.h"

Buff::Buff() { SPDLOG_TRACE("Constructed."); }

Buff::~Buff() { SPDLOG_TRACE("Destructed."); }

void Buff::Init() {
  center_ = cv::Point2f(0, 0);
  team_ = game::Team::kUNKNOWN;
  direction_ = rotation::Direction::kUNKNOWN;
}

std::vector<Armor> Buff::GetArmors() {
  SPDLOG_DEBUG("armors_: {}", armors_.size());
  return armors_;
}

void Buff::SetArmors(std::vector<Armor> armors) { armors_ = armors; }

cv::Point2f Buff::GetCenter() {
  SPDLOG_DEBUG("center_: {}, {}", center_.x, center_.y);
  return center_;
}

void Buff::SetCenter(cv::Point2f center) { center_ = center; }

rotation::Direction Buff::GetDirection() { return direction_; }

void Buff::SetDirection(rotation::Direction direction) {
  std::string rot;
  switch (direction_) {
    case rotation::Direction::kCLOCKWISE:
      rot = "CLOCKWISE";
      break;
    case rotation::Direction::kANTI:
      rot = "ANTICLOCKWISE";
      break;
    default:
      rot = "UNKNOWN";
      break;
  }
  SPDLOG_DEBUG("direction_: {}", rot);
  direction_ = direction;
}

double Buff::GetSpeed() {
  SPDLOG_DEBUG("speed_: {}", speed_);
  return speed_;
}

void Buff::SetSpeed(double speed) { speed_ = speed; }

Armor Buff::GetTarget() {
  SPDLOG_DEBUG("Got it.");
  return target_;
}

void Buff::SetTarget(Armor target) { target_ = target; }

Armor Buff::GetPredict() {
  SPDLOG_DEBUG("Got it.");
  return predict_;
}

void Buff::SetPridict(Armor target) { target_ = target; }

game::Team Buff::GetTeam() {
  SPDLOG_DEBUG("team_: {}", team_);
  return team_;
}

void Buff::SetTeam(game::Team team) { team_ = team; }

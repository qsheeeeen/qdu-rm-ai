#include "buff.hpp"

#include "opencv2/opencv.hpp"
#include "spdlog/spdlog.h"

Buff::Buff() : center_(cv::Point2f(0, 0)) { SPDLOG_TRACE("Constructed."); }

Buff::Buff(game::Team team) : center_(cv::Point2f(0, 0)) {
  SetTeam(team);
  SPDLOG_TRACE("Constructed.");
}

Buff::~Buff() { SPDLOG_TRACE("Destructed."); }

void Buff::SetTeam(game::Team team) { team_ = team; }

tbb::concurrent_vector<Armor> Buff::GetArmors() {
  SPDLOG_DEBUG("armors_: {}", armors_.size());
  return armors_;
}

void Buff::SetArmors(tbb::concurrent_vector<Armor> armors) { armors_ = armors; }

cv::Point2f Buff::GetCenter() {
  SPDLOG_DEBUG("center_: {}, {}", center_.x, center_.y);
  return center_;
}

void Buff::SetCenter(cv::Point2f center) { center_ = center; }

common::Direction Buff::GetDirection() { return direction_; }

void Buff::SetDirection(common::Direction direction) {
  SPDLOG_DEBUG("direction_: {}", common::DirectionToString(direction));
  direction_ = direction;
}

double Buff::GetTime() {
  SPDLOG_DEBUG("time_: {}", time_);
  return time_++;
}

void Buff::SetTime(double speed) { time_ = speed; }

Armor Buff::GetTarget() { return target_; }

void Buff::SetTarget(Armor target) { target_ = target; }

Armor Buff::GetPredict() { return predict_; }

void Buff::SetPridict(Armor predict) { predict_ = predict; }

game::Team Buff::GetTeam() {
  SPDLOG_DEBUG("team_: {}", team_);
  return team_;
}

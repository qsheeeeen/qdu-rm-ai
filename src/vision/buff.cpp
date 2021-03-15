#include "buff.hpp"

#include "opencv2/opencv.hpp"
#include "spdlog/spdlog.h"

Buff::Buff() { SPDLOG_TRACE("Constructed."); }

Buff::~Buff() { SPDLOG_TRACE("Destructed."); }

std::vector<Armor> Buff::GetArmors() {
  SPDLOG_DEBUG("armors_: {}", armors_.size());
  return armors_;
}

void Buff::SetArmors(std::vector<Armor> armors) { armors_ = armors; }

cv::RotatedRect Buff::GetCenter() {
  SPDLOG_DEBUG("center_: {}, {}", center_.center.x, center_.center.y);
  return center_;
}

void Buff::SetCenter(cv::RotatedRect center) { center_ = center; }

double Buff::GetSpeed() {
  SPDLOG_DEBUG("rects_: {}", speed_);
  return speed_;
}

void Buff::SetSpeed(double time) { speed_ = 0.785 * sin(1.884 * time) + 1.305; }

Armor Buff::GetTarget() {
  SPDLOG_DEBUG("Got it.");
  return target_;
}

void Buff::SetTarget(Armor target) { target_ = target; }

std::vector<cv::RotatedRect> Buff::GetTracks() {
  SPDLOG_DEBUG("rects_: {}", tracks_.size());
  return tracks_;
}

void Buff::SetTracks(std::vector<cv::RotatedRect> tracks) { tracks_ = tracks; }

game::Team Buff::GetTeam() {
  SPDLOG_DEBUG("team_: {}", team_);
  return team_;
}

void Buff::SetTeam(game::Team team) { team_ = team; }

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

cv::Point2f Buff::GetCenter() {
  SPDLOG_DEBUG("center_: {}, {}", center_.x, center_.y);
  return center_;
}

void Buff::SetCenter(cv::Point2f center) { center_ = center; }

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

game::Team Armor::GetTeam() {
  SPDLOG_DEBUG("team_: {}", team_);
  return team_;
}

void Armor::SetTeam(game::Team team) { team_ = team; }

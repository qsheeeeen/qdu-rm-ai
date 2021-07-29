#pragma once

#include <vector>

#include "common.hpp"
#include "light_bar.hpp"
#include "object.hpp"
#include "opencv2/opencv.hpp"

class Armor : public ImageObject, public PhysicObject {
 private:
  game::Team team_ = game::Team::kUNKNOWN;
  common::Euler aiming_euler_;

 public:
  Armor();
  Armor(const LightBar &left_bar, const LightBar &right_bar);
  Armor(const cv::RotatedRect &rect);
  ~Armor();

  game::Team GetTeam() const;
  void SetTeam(game::Team team);

  common::Euler GetAimEuler() const;
  void SetAimEuler(const common::Euler &elur);
};

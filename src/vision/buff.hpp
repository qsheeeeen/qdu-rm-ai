#pragma once

#include <vector>

#include "armor.hpp"
#include "common.hpp"

class Buff {
 private:
  cv::Point2f center_;
  std::vector<Armor> armors_;
  Armor target_;
  game::Team team_;

 public:
  Buff();
  Buff(game::Team team);
  Buff(const cv::Point2f &center, const std::vector<Armor> &armors,
       const Armor &target, game::Team team = game::Team::kUNKNOWN);
  ~Buff();

  const std::vector<Armor> &GetArmors() const;
  void SetArmors(const std::vector<Armor> &armors);

  const cv::Point2f &GetCenter() const;
  void SetCenter(const cv::Point2f &center);

  const Armor &GetTarget() const;
  void SetTarget(const Armor &target);

  const game::Team &GetTeam() const;
  void SetTeam(const game::Team &team);
};
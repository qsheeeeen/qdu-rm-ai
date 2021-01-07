#pragma once

#include <vector>

#include "armor.hpp"
#include "game.hpp"
#include "light_bar.hpp"

class ArmorDetector {
 private:
  game::Team enemy_team_;

 public:
  ArmorDetector();
  ArmorDetector(game::Team enemy_team);
  ~ArmorDetector();

  void Init(game::Team enemy_team);

  void StoreParams(std::string path);
  void LoadParams(std::string path);

  std::vector<LighjtBar> FindLightBars(cv::Mat frame);
  std::vector<Armor> MatchLightBars(std::vector<LighjtBar> lightbars);
};

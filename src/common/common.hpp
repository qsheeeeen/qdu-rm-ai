#pragma once

#include <string>

namespace common {

struct Euler {
  double yaw, pitch, roll;
};

}  // namespace common

namespace game {

enum class Team {
  kUNKNOWN,
  kDEAD,
  kBLUE,
  kRED,
};

enum class Model {
  kUNKNOWN,
  kINFANTRY,
  kHERO,
  kENGINEER,
  kDRONE,
  kSENTRY,
  kBASE,
  kOUTPOST,
};

std::string TeamToString(Team team);
std::string ModelToString(Model model);
Model StringToModel(std::string name);

bool HasBigArmor(Model model);

}  // namespace game
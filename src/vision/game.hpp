#pragma once

#include <string>

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
};

std::string TeamToString(Team team);
std::string ModelToString(Model model);

bool HasBigArmor(Model model);

}  // namespace game
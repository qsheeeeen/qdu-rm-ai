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

std::string TeamToString(Team team) {
  switch (team) {
    case Team::kUNKNOWN:
      return std::string("Unknown");
    case Team::kDEAD:
      return std::string("Dead");
    case Team::kBLUE:
      return std::string("Blue");
    case Team::kRED:
      return std::string("Red");
    default:
      return std::string("Unknown");
  }
}

std::string ModelToString(Model model) {
  switch (model) {
    case Model::kUNKNOWN:
      return std::string("Unknown");
    case Model::kINFANTRY:
      return std::string("Infantry");
    case Model::kHERO:
      return std::string("Hero");
    case Model::kENGINEER:
      return std::string("Engineer");
    case Model::kDRONE:
      return std::string("Drone");
    case Model::kSENTRY:
      return std::string("Sentry");
    case Model::kBASE:
      return std::string("Base");
    default:
      return std::string("Unknown");
  }
}

}  // namespace game
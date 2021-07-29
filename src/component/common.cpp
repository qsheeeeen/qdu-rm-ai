#include "common.hpp"

#include <algorithm>
#include <cctype>
#include <string>

namespace component {

std::string DirectionToString(Direction direction) {
  switch (direction) {
    case Direction::kUNKNOWN:
      return std::string("Unknown");
    case Direction::kCW:
      return std::string("Clockwise");
    case Direction::kCCW:
      return std::string("Anticlockwise");
    default:
      return std::string("Unknown");
  }
}

}  // namespace component

namespace game {

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
    case Model::kOUTPOST:
      return std::string("Outpost");
    default:
      return std::string("Unknown");
  }
}

Model StringToModel(std::string name) {
  std::transform(name.begin(), name.end(), name.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (!name.compare("infantry") || !name.compare("3") || !name.compare("4") ||
      !name.compare("5")) {
    return Model::kINFANTRY;
  }
  if (!name.compare("hero") || !name.compare("1")) {
    return Model::kHERO;
  }
  if (!name.compare("engineer") || !name.compare("2")) {
    return Model::kENGINEER;
  }
  if (!name.compare("frone")) {
    return Model::kDRONE;
  }
  if (!name.compare("sentry")) {
    return Model::kSENTRY;
  }
  if (!name.compare("base")) {
    return Model::kBASE;
  }
  if (!name.compare("outpost")) {
    return Model::kOUTPOST;
  }
  return Model::kUNKNOWN;
}

bool HasBigArmor(Model model) {
  return (model == Model::kHERO || model == Model::kSENTRY);
}

}  // namespace game

namespace algo {

double RelativeDifference(double a, double b) {
  double diff = std::abs(a - b);
  double base = std::max(std::abs(a), std::abs(b));
  return diff / base;
}

}  // namespace algo

#include <vector>

#include "armor.hpp"

namespace rotation {

enum class Direction { kUNKNOWN, kCLOCKWISE, kANTI };

}  // namespace rotation

class Buff {
 private:
  cv::RotatedRect center_;
  std::vector<Armor> armors_;
  Armor target_;
  std::vector<cv::RotatedRect> tracks_;
  game::Team team_ = game::Team::kUNKNOWN;
  double speed_;
  rotation::Direction direction_ = rotation::Direction::kUNKNOWN;

 public:
  Buff();
  ~Buff();

  std::vector<Armor> GetArmors();
  void SetArmors(std::vector<Armor> armors);

  cv::RotatedRect GetCenter();
  void SetCenter(cv::RotatedRect center);

  double GetSpeed();
  void SetSpeed(double time);

  Armor GetTarget();
  void SetTarget(Armor target);

  std::vector<cv::RotatedRect> GetTracks();
  void SetTracks(std::vector<cv::RotatedRect> tracks);

  game::Team GetTeam();
  void SetTeam(game::Team team);
};
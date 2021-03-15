#include <vector>

#include "armor.hpp"

class Buff {
 private:
  cv::RotatedRect center_;
  std::vector<Armor> armors_;
  Armor target_;
  std::vector<cv::RotatedRect> tracks_;
  game::Team team_ = game::Team::kUNKNOWN;
  double speed_, time_;

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
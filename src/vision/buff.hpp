#include <vector>

#include "armor.hpp"

class Buff {
 private:
  cv::Point2f center_;
  std::vector<Armor> armors_;
  Armor target_;
  std::vector<cv::RotatedRect> tracks_;
  game::Team team_ = game::Team::kUNKNOWN;
  cv::Point3f world_coord_;

 public:
  Buff();
  ~Buff();

  std::vector<Armor> GetArmors();
  void SetArmors(std::vector<Armor> armors);

  cv::Point2f GetCenter();
  void SetCenter(cv::Point2f center);

  Armor GetTarget();
  void SetTarget(Armor target);

  std::vector<cv::RotatedRect> GetTracks();
  void SetTracks(std::vector<cv::RotatedRect> tracks);

  game::Team GetTeam();
  void SetTeam(game::Team team);
};
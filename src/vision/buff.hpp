#include <vector>

#include "armor.hpp"
#include "track.hpp"

class Buff {
 private:
  cv::Point2f center_;
  std::vector<Armor> armors_;
  Armor target_, predict_;
  game::Team team_;
  double time_;
  rotation::Direction direction_;

 public:
  Buff();
  ~Buff();

  void Init();

  std::vector<Armor> GetArmors();
  void SetArmors(std::vector<Armor> armors);

  cv::Point2f GetCenter();
  void SetCenter(cv::Point2f center);

  rotation::Direction GetDirection();
  void SetDirection(rotation::Direction direction);

  double GetTime();
  void SetTime(double time);

  Armor GetTarget();
  void SetTarget(Armor target);

  Armor GetPredict();
  void SetPridict(Armor target);

  game::Team GetTeam();
  void SetTeam(game::Team team);
};
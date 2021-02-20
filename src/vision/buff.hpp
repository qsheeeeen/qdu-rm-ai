#include <vector>

#include "armor.hpp"

class Buff {
 private:
  std::vector<std::vector<cv::Point2f>> contours_;
  std::vector<Armor> armors_;
  std::vector<cv::RotatedRect> tracks_;
  game::Team team_ = game::Team::kUNKNOWN;
  cv::Point3f world_coord_;

 public:
  Buff();
  Buff(std::vector<std::vector<cv::Point2f>> contours);
  ~Buff();

  void Init();
  bool IsTarget();

  const cv::Point2f &Center2D(cv::RotatedRect rect);
  std::vector<cv::Point2f> Vertices2D(cv::RotatedRect rect);

  std::vector<Armor> GetArmors();
  void SetArmors(std::vector<Armor> armors);

  std::vector<std::vector<cv::Point2f>> GetContours();
  void SetContours(std::vector<std::vector<cv::Point2f>> contours);

  std::vector<cv::RotatedRect> GetTracks();
  void SetTracks(std::vector<cv::RotatedRect> tracks);

  game::Team GetTeam();
  void SetTeam(game::Team team);
};
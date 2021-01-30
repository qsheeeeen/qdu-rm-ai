#pragma once

#include <chrono>
#include <vector>

#include "armor.hpp"
#include "robot.hpp"

struct RobotDetectorParam {
  double width_diff_th;
  double height_diff_th;
  double center_dist_low_th;
  double center_dist_high_th;
  double axis_angle_th;
};

class RobotDetector {
 private:
  cv::Size frame_size_;
  RobotDetectorParam params_;

  cv::Mat cam_mat_, distor_coff_;

  std::vector<Robot> robots_;

  std::chrono::milliseconds duration_robots_;

  void InitDefaultParams(const std::string& path);
  bool PrepareParams(const std::string& path);
  void LoadCameraMat(const std::string& path);

  void Estimate3D(Armor& armor);
  double AxisAngle(cv::Vec3d& axis1, cv::Vec3d& axis2);
  void MatchArmors(Armor& armor);

 public:
  RobotDetector();
  RobotDetector(const std::string& params_path,
                const std::string& cam_mat_path);
  ~RobotDetector();

  void Init(const std::string& params_path, const std::string& cam_mat_path);

  void Detect(std::vector<Armor>& armors);
  void VisualizeResult(cv::Mat& output, bool add_lable = true);
};

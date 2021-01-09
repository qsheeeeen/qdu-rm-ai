#include "armor_detector.hpp"

#include <ostream>

#include "opencv2/opencv.hpp"
#include "spdlog/spdlog.h"

void ArmorDetector::FindLightBars(const cv::Mat &frame) {
  lightbars_.clear();
  armors_.clear();

  cv::Mat channels[3];
  cv::split(frame, channels);

  cv::Mat result;
  if (enemy_team_ == game::Team::kBLUE) {
    result = channels[0] - channels[2];
  } else if (enemy_team_ == game::Team::kRED) {
    result = channels[2] - channels[0];
  }

  // TODO: sharpen blur add contrast.

  cv::adaptiveThreshold(result, result, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv::THRESH_BINARY, int(params_["threshold_block"]), 0.);

  const float erosion_size = float(params_["erosion_size"]);
  cv::Mat kernel = cv::getStructuringElement(
      cv::MORPH_RECT, cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
      cv::Point(erosion_size, erosion_size));

  cv::morphologyEx(result, result, cv::MORPH_OPEN, kernel);

  std::vector<std::vector<cv::Point> > contours;
  cv::findContours(frame, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

  for (const auto &contour : contours) {
    if (contour.size() < float(params_["contour_size_th"])) continue;

    const double area = cv::contourArea(contour);
    if (area < float(params_["contour_area_low_th"]) &&
        area > float(params_["contour_area_high_th"]))
      continue;

    auto potential_bar = LightBar(cv::minAreaRect(contour));

    if (std::abs(potential_bar.Angle()) > float(params_["bar_angle_high_th"]))
      continue;

    if (potential_bar.AspectRatio() < float(params_["aspect_ratio_low_th"]) &&
        potential_bar.AspectRatio() > float(params_["aspect_ratio_high_th"]))
      continue;

    // TODO: Add more rules.
    lightbars_.emplace_back(potential_bar);
  }

  std::sort(lightbars_.begin(), lightbars_.end(),
            [](LightBar &bar1, LightBar &bar2) {
              return bar1.Center().x < bar2.Center().x;
            });
}

void ArmorDetector::MatchLightBars() {
  for (auto iti = lightbars_.begin(); iti != lightbars_.end(); ++iti) {
    for (auto itj = iti + 1; itj != lightbars_.end(); ++itj) {
      float angle_diff = std::abs(iti->Angle() - itj->Angle()) /
                         (iti->Angle() + itj->Angle()) * 2.0;
      if (angle_diff > float(params_["angle_diff_th"])) continue;

      float length_diff = std::abs(iti->Length() - itj->Length()) /
                          (iti->Length() + itj->Length()) * 2.0;
      if (length_diff > float(params_["length_diff_th"])) continue;

      float height_diff = std::abs(iti->Center().y - itj->Center().y) /
                          (iti->Center().y + itj->Center().y) * 2.0;
      if (height_diff > float(params_["height_diff_th"])) continue;

      float center_dist = cv::norm(iti->Center() - itj->Center());
      if (center_dist > float(params_["center_dist_th"])) continue;

      float area_diff = std::abs(iti->Area() - itj->Area()) /
                        (iti->Area() + itj->Area()) * 2.0;
      if (area_diff > float(params_["area_diff_th"])) continue;

      auto armor = Armor(*iti, *itj);
      armor.SetModel(armor_classifier_.Classify(armor));
      armors_.emplace_back(armor);
    }
  }
}

game::Model ArmorDetector::GetModel(const Armor &armor) {
  return armor_classifier_.Classify(armor);
}

void ArmorDetector::VisualizeLightBar(cv::Mat &output, bool add_lable) {
  if (!lightbars_.empty()) {
    for (auto &bar : lightbars_) {
      auto vertices = bar.Vertices();
      for (size_t i = 0; i < vertices.size(); ++i)
        cv::line(output, vertices[i], vertices[(i + 1) % 4], green_);

      cv::circle(output, bar.Center(), 2, green_);

      if (add_lable) {
        std::ostringstream buf;
        buf << bar.Center().x << ", " << bar.Center().y;
        cv::putText(output, buf.str(), bar.Center(),
                    cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0, green_);
      }
    }
  }
}

void ArmorDetector::VisualizeArmor(cv::Mat &output, bool add_lable) {
  if (!armors_.empty()) {
    for (auto &armor : armors_) {
      auto vertices = armor.Vertices();
      for (size_t i = 0; i < vertices.size(); ++i)
        cv::line(output, vertices[i], vertices[(i + 1) % 4], green_);

      cv::circle(output, armor.Center(), 2, green_);

      if (add_lable) {
        std::ostringstream buf;
        buf << armor.Center().x << ", " << armor.Center().y;
        cv::putText(output, buf.str(), armor.Center(),
                    cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0, green_);
      }
    }
  }
}

ArmorDetector::ArmorDetector() { SPDLOG_DEBUG("[ArmorDetector] Constructed."); }

ArmorDetector::ArmorDetector(std::string params_path, game::Team enemy_team) {
  Init(params_path, enemy_team);
  SPDLOG_DEBUG("[ArmorDetector] Constructed.");
}

ArmorDetector::~ArmorDetector() {
  params_.release();
  SPDLOG_DEBUG("[ArmorDetector] Destructed.");
}

void ArmorDetector::Init(std::string params_path, game::Team enemy_team) {
  params_.open(params_path, cv::FileStorage::READ | cv::FileStorage::MEMORY |
                                cv::FileStorage::FORMAT_JSON);
  if (!params_.isOpened()) InitDefaultParams(params_path);
  enemy_team_ = enemy_team;
  SPDLOG_DEBUG("[ArmorDetector] Inited.");
}

void ArmorDetector::InitDefaultParams(std::string params_path) {
  params_.open(params_path, cv::FileStorage::WRITE | cv::FileStorage::MEMORY |
                                cv::FileStorage::FORMAT_JSON);

  params_.startWriteStruct("preprocessing", cv::FileNode::SEQ);
  params_ << "threshold_block" << 7;
  params_ << "erosion_size " << 2;
  params_.endWriteStruct();

  params_.startWriteStruct("light_bars_find", cv::FileNode::SEQ);
  params_ << "contour_size_th" << 6;
  params_ << "contour_area_low_th" << 10;
  params_ << "contour_area_high_th" << INFINITY;
  params_ << "bar_angle_high_th" << 60;
  params_ << "aspect_ratio_low_th" << 1;
  params_ << "aspect_ratio_high_th" << 2;
  params_.endWriteStruct();

  params_.startWriteStruct("light_bars_find", cv::FileNode::SEQ);
  params_ << "angle_diff_th" << 0.5;
  params_ << "length_diff_th" << 0.5;
  params_ << "height_diff_th" << 0.5;
  params_ << "center_dist_th" << 0.5;
  params_ << "area_diff_th" << 0.5;
  params_.endWriteStruct();
}

void ArmorDetector::Detect(cv::Mat &frame) {
  FindLightBars(frame);
  MatchLightBars();
}

void ArmorDetector::VisualizeResult(cv::Mat &output, bool draw_bars,
                                    bool draw_armor, bool add_lable) {
  if (draw_bars) VisualizeLightBar(output, add_lable);
  if (draw_armor) VisualizeArmor(output, add_lable);
}

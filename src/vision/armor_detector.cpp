#include "armor_detector.hpp"

#include <ostream>

#include "opencv2/opencv.hpp"
#include "spdlog/spdlog.h"

void ArmorDetector::InitDefaultParams(std::string params_path) {
  fs_.open(params_path, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);

  fs_ << "binary_th" << 230;
  fs_ << "erosion_size" << 5;

  fs_ << "contour_size_th" << 20;
  fs_ << "contour_area_low_th" << 100;
  fs_ << "contour_area_high_th" << 10000;
  fs_ << "bar_area_low_th" << 100;
  fs_ << "bar_area_high_th" << 10000;
  fs_ << "angle_high_th" << 60;
  fs_ << "aspect_ratio_low_th" << 2;
  fs_ << "aspect_ratio_high_th" << 5;

  fs_ << "angle_diff_th" << 0.5;
  fs_ << "length_diff_th" << 0.5;
  fs_ << "height_diff_th" << 0.5;
  fs_ << "area_diff_th" << 0.5;
  fs_ << "center_dist_low_th" << 1.5;
  fs_ << "center_dist_high_th" << 5;
  fs_.release();

  SPDLOG_DEBUG("Inited params.");
}

void ArmorDetector::PrepareParams() {
  params_.binary_th = fs_["binary_th"];
  params_.erosion_size = fs_["erosion_size"];

  params_.contour_size_th = int(fs_["contour_size_th"]);
  params_.contour_area_low_th = fs_["contour_area_low_th"];
  params_.contour_area_high_th = fs_["contour_area_high_th"];
  params_.bar_area_low_th = fs_["bar_area_low_th"];
  params_.bar_area_high_th = fs_["bar_area_high_th"];
  params_.angle_high_th = fs_["angle_high_th"];
  params_.aspect_ratio_low_th = fs_["aspect_ratio_low_th"];
  params_.aspect_ratio_high_th = fs_["aspect_ratio_high_th"];

  params_.angle_diff_th = fs_["angle_diff_th"];
  params_.length_diff_th = fs_["length_diff_th"];
  params_.height_diff_th = fs_["height_diff_th"];
  params_.area_diff_th = fs_["area_diff_th"];
  params_.center_dist_low_th = fs_["center_dist_low_th"];
  params_.center_dist_high_th = fs_["center_dist_high_th"];
}

void ArmorDetector::FindLightBars(const cv::Mat &frame) {
  lightbars_.clear();
  armors_.clear();

  cv::Mat channels[3];
  cv::split(frame, channels);

  cv::Mat result = channels[0];
  // if (enemy_team_ == game::Team::kBLUE) {
  //   result = channels[0] - channels[2];
  // } else if (enemy_team_ == game::Team::kRED) {
  //   result = channels[2] - channels[0];
  // }

  // TODO: sharpen blur add contrast.

  cv::threshold(result, result, params_.binary_th, 255., cv::THRESH_BINARY);

  const int erosion_size = params_.erosion_size;
  cv::Mat kernel = cv::getStructuringElement(
      cv::MORPH_RECT, cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1));
  cv::morphologyEx(result, result, cv::MORPH_OPEN, kernel);

  std::vector<std::vector<cv::Point> > contours;
  cv::findContours(result, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
  SPDLOG_INFO("Found contours: {}", contours.size());

  for (const auto &contour : contours) {
    // TODO: computional light first.
    if (contour.size() < params_.contour_size_th) continue;

    const double c_area = cv::contourArea(contour);
    if (c_area < params_.contour_area_low_th) continue;
    if (c_area > params_.contour_area_high_th) continue;

    auto potential_bar = LightBar(cv::minAreaRect(contour));

    const double b_area = potential_bar.Area();
    if (b_area < params_.bar_area_low_th) continue;
    if (b_area > params_.bar_area_high_th) continue;

    if (std::abs(potential_bar.Angle()) > double(params_.angle_high_th))
      continue;

    const double ar = potential_bar.AspectRatio();
    if (ar < params_.aspect_ratio_low_th) continue;
    if (ar > params_.aspect_ratio_high_th) continue;

    // TODO: Add more rules.
    lightbars_.emplace_back(potential_bar);
  }

  SPDLOG_INFO("Found light bars: {}", lightbars_.size());

  std::sort(lightbars_.begin(), lightbars_.end(),
            [](LightBar &bar1, LightBar &bar2) {
              return bar1.Center().x < bar2.Center().x;
            });
}

void ArmorDetector::MatchLightBars() {
  for (auto iti = lightbars_.begin(); iti != lightbars_.end(); ++iti) {
    for (auto itj = iti + 1; itj != lightbars_.end(); ++itj) {
      if (iti->Angle() > 0.) {
        if (iti->Center().y > itj->Center().y) continue;
      } else {
        if (iti->Center().y < itj->Center().y) continue;
      }

      const double angle_diff = std::abs(iti->Angle() - itj->Angle());
      if (angle_diff > params_.angle_diff_th) continue;

      double length_diff =
          std::abs(iti->Length() - itj->Length()) / iti->Length();
      if (length_diff > params_.length_diff_th) continue;

      // TODO: reletive to img size;
      double height_diff = std::abs(iti->Center().y - itj->Center().y);
      if (height_diff > params_.height_diff_th) continue;

      double area_diff = std::abs(iti->Area() - itj->Area()) / iti->Area();
      if (area_diff > params_.area_diff_th) continue;

      double center_dist = cv::norm(iti->Center() - itj->Center());
      double l = iti->Length();
      if (center_dist < l * params_.center_dist_low_th) continue;
      if (center_dist > l * params_.center_dist_high_th) continue;

      auto armor = Armor(*iti, *itj);
      armor.SetModel(armor_classifier_.Classify(armor));
      armors_.emplace_back(armor);
      break;
    }
  }
  SPDLOG_INFO("Found armors: {}", armors_.size());
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

ArmorDetector::ArmorDetector() { SPDLOG_DEBUG("Constructed."); }

ArmorDetector::ArmorDetector(std::string params_path, game::Team enemy_team) {
  Init(params_path, enemy_team);
  SPDLOG_DEBUG("Constructed.");
}

ArmorDetector::~ArmorDetector() {
  fs_.release();
  SPDLOG_DEBUG("Destructed.");
}

void ArmorDetector::Init(std::string params_path, game::Team enemy_team) {
  fs_.open(params_path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);

  if (fs_.isOpened()) {
    PrepareParams();
  } else {
    InitDefaultParams(params_path);
    fs_.open(params_path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    PrepareParams();
    SPDLOG_WARN("Can not find parasm file. Created and reloaded.");
  }
  enemy_team_ = enemy_team;
  SPDLOG_DEBUG("Inited.");
}

void ArmorDetector::Detect(cv::Mat &frame) {
  SPDLOG_DEBUG("Detecting");
  FindLightBars(frame);
  MatchLightBars();
  SPDLOG_DEBUG("Detected.");
}

void ArmorDetector::VisualizeResult(cv::Mat &output, bool draw_bars,
                                    bool draw_armor, bool add_lable) {
  if (draw_bars) VisualizeLightBar(output, add_lable);
  if (draw_armor) VisualizeArmor(output, add_lable);
}

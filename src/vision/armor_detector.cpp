#include "armor_detector.hpp"

#include <ostream>

#include "opencv2/opencv.hpp"
#include "spdlog/spdlog.h"

namespace {

const auto kCV_FONT = cv::FONT_HERSHEY_SIMPLEX;
const auto kGREEN = cv::Scalar(0., 255., 0.);
const auto kRED = cv::Scalar(0., 0., 255.);
const auto kYELLOW = cv::Scalar(0., 255., 255.);

}  // namespace

void ArmorDetector::InitDefaultParams(const std::string &params_path) {
  cv::FileStorage fs(params_path,
                     cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);

  fs << "binary_th" << 220;
  fs << "se_erosion" << 5;
  fs << "ap_erosion" << 1.;

  fs << "contour_size_low_th" << 20;
  fs << "contour_area_low_th" << 100;
  fs << "contour_area_high_th" << 10000;
  fs << "bar_area_low_th" << 100;
  fs << "bar_area_high_th" << 10000;
  fs << "angle_high_th" << 60;
  fs << "aspect_ratio_low_th" << 2;
  fs << "aspect_ratio_high_th" << 6;

  fs << "angle_diff_th" << 10;
  fs << "length_diff_th" << 0.2;
  fs << "height_diff_th" << 0.2;
  fs << "area_diff_th" << 0.6;
  fs << "center_dist_low_th" << 1;
  fs << "center_dist_high_th" << 4;
  SPDLOG_DEBUG("Inited params.");
}

bool ArmorDetector::PrepareParams(const std::string &params_path) {
  cv::FileStorage fs(params_path,
                     cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
  if (fs.isOpened()) {
    params_.binary_th = fs["binary_th"];
    params_.se_erosion = fs["se_erosion"];
    params_.ap_erosion = fs["ap_erosion"];

    params_.contour_size_low_th = int(fs["contour_size_low_th"]);
    params_.contour_area_low_th = fs["contour_area_low_th"];
    params_.contour_area_high_th = fs["contour_area_high_th"];
    params_.bar_area_low_th = fs["bar_area_low_th"];
    params_.bar_area_high_th = fs["bar_area_high_th"];
    params_.angle_high_th = fs["angle_high_th"];
    params_.aspect_ratio_low_th = fs["aspect_ratio_low_th"];
    params_.aspect_ratio_high_th = fs["aspect_ratio_high_th"];

    params_.angle_diff_th = fs["angle_diff_th"];
    params_.length_diff_th = fs["length_diff_th"];
    params_.height_diff_th = fs["height_diff_th"];
    params_.area_diff_th = fs["area_diff_th"];
    params_.center_dist_low_th = fs["center_dist_low_th"];
    params_.center_dist_high_th = fs["center_dist_high_th"];
    return true;
  } else {
    SPDLOG_ERROR("Can not load params.");
    return false;
  }
}

void ArmorDetector::FindLightBars(const cv::Mat &frame) {
  const auto start = std::chrono::high_resolution_clock::now();
  lightbars_.clear();
  targets_.clear();

  frame_size_ = cv::Size(frame.cols, frame.rows);

  cv::Mat channels[3], result;
  cv::split(frame, channels);

#if 1
  if (enemy_team_ == game::Team::kBLUE) {
    result = channels[0];
  } else if (enemy_team_ == game::Team::kRED) {
    result = channels[2];
  }
#else
  if (enemy_team_ == game::Team::kBLUE) {
    result = channels[0] - channels[2];
  } else if (enemy_team_ == game::Team::kRED) {
    result = channels[2] - channels[0];
  }
#endif

  cv::threshold(result, result, params_.binary_th, 255., cv::THRESH_BINARY);

  cv::Mat kernel = cv::getStructuringElement(
      cv::MORPH_ELLIPSE,
      cv::Size(2 * params_.se_erosion + 1, 2 * params_.se_erosion + 1));

  cv::morphologyEx(result, result, cv::MORPH_OPEN, kernel);
  cv::findContours(result, contours_, cv::RETR_LIST,
                   cv::CHAIN_APPROX_TC89_KCOS);

#if 0
  contours_poly_.resize(contours_.size());
  for (size_t k = 0; k < contours_.size(); ++k) {
    cv::approxPolyDP(cv::Mat(contours_[k]), contours_poly_[k],
                     params_.ap_erosion, true);
  }
#endif

  SPDLOG_DEBUG("Found contours: {}", contours_.size());

  for (const auto &contour : contours_) {
    if (contour.size() < params_.contour_size_low_th) continue;

    const double c_area = cv::contourArea(contour);
    if (c_area < params_.contour_area_low_th) continue;
    if (c_area > params_.contour_area_high_th) continue;

    auto potential_bar = LightBar(cv::minAreaRect(contour));

    if (std::abs(potential_bar.Angle()) > params_.angle_high_th) continue;

    const double b_area = potential_bar.Area();
    if (b_area < params_.bar_area_low_th) continue;
    if (b_area > params_.bar_area_high_th) continue;

    const double ar = potential_bar.AspectRatio();
    if (ar < params_.aspect_ratio_low_th) continue;
    if (ar > params_.aspect_ratio_high_th) continue;

    lightbars_.emplace_back(potential_bar);
  }

  SPDLOG_DEBUG("Found light bars: {}", lightbars_.size());

  std::sort(lightbars_.begin(), lightbars_.end(),
            [](LightBar &bar1, LightBar &bar2) {
              return bar1.Center().x < bar2.Center().x;
            });
  const auto stop = std::chrono::high_resolution_clock::now();
  duration_bars_ =
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

  SPDLOG_DEBUG("duration_bars_: {} ms", duration_bars_.count());
}

void ArmorDetector::MatchLightBars(const cv::Mat &frame) {
  const auto start = std::chrono::high_resolution_clock::now();
  for (auto iti = lightbars_.begin(); iti != lightbars_.end(); ++iti) {
    for (auto itj = iti + 1; itj != lightbars_.end(); ++itj) {
      const double angle_diff =
          std::abs(iti->Angle() - itj->Angle()) / iti->Angle();
      if (angle_diff > params_.angle_diff_th) continue;

      const double length_diff =
          std::abs(iti->Length() - itj->Length()) / iti->Length();
      if (length_diff > params_.length_diff_th) continue;

      const double height_diff = std::abs(iti->Center().y - itj->Center().y);
      if (height_diff > (params_.height_diff_th * frame_size_.height)) continue;

      const double area_diff =
          std::abs(iti->Area() - itj->Area()) / iti->Area();
      if (area_diff > params_.area_diff_th) continue;

      const double center_dist = cv::norm(iti->Center() - itj->Center());
      const double l = iti->Length();
      if (center_dist < l * params_.center_dist_low_th) continue;
      if (center_dist > l * params_.center_dist_high_th) continue;

      auto armor = Armor(*iti, *itj);
      armor_classifier_.ClassifyModel(armor, frame);
      armor_classifier_.ClassifyTeam(armor, frame);
      targets_.emplace_back(armor);
      break;
    }
  }
  SPDLOG_DEBUG("Found armors: {}", targets_.size());

  const auto stop = std::chrono::high_resolution_clock::now();
  duration_armors_ =
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  SPDLOG_DEBUG("duration_armors_: {} ms", duration_armors_.count());
}

void ArmorDetector::VisualizeLightBar(const cv::Mat &output, bool add_lable) {
  if (!lightbars_.empty()) {
    for (auto &bar : lightbars_) {
      auto vertices = bar.Vertices();
      for (std::size_t i = 0; i < vertices.size(); ++i)
        cv::line(output, vertices[i], vertices[(i + 1) % vertices.size()],
                 kGREEN);

      cv::drawMarker(output, bar.Center(), kGREEN, cv::MARKER_CROSS);

      if (add_lable) {
        std::ostringstream buf;
        buf << bar.Center().x << ", " << bar.Center().y;
        cv::putText(output, buf.str(), vertices[1], kCV_FONT, 1.0, kGREEN);
      }
    }
  }
}

void ArmorDetector::VisualizeArmor(const cv::Mat &output, bool add_lable) {
  if (!targets_.empty()) {
    for (auto &armor : targets_) {
      auto vertices = armor.SurfaceVertices();
      for (std::size_t i = 0; i < vertices.size(); ++i) {
        cv::line(output, vertices[i], vertices[(i + 1) % 4], kGREEN);
      }
      cv::drawMarker(output, armor.SurfaceCenter(), kGREEN, cv::MARKER_DIAMOND);

      if (add_lable) {
        std::ostringstream buf;
        buf << armor.SurfaceCenter().x << ", " << armor.SurfaceCenter().y;
        cv::putText(output, buf.str(), vertices[1], kCV_FONT, 1.0, kGREEN);
      }
    }
  }
}

ArmorDetector::ArmorDetector() { SPDLOG_TRACE("Constructed."); }

ArmorDetector::ArmorDetector(const std::string &params_path,
                             game::Team enemy_team) {
  LoadParams(params_path);
  SetEnemyTeam(enemy_team);
  SPDLOG_TRACE("Constructed.");
}

ArmorDetector::~ArmorDetector() { SPDLOG_TRACE("Destructed."); }

void ArmorDetector::SetEnemyTeam(game::Team enemy_team) {
  enemy_team_ = enemy_team;
}

const std::vector<Armor> &ArmorDetector::Detect(const cv::Mat &frame) {
  SPDLOG_DEBUG("Detecting");
  FindLightBars(frame);
  MatchLightBars(frame);
  SPDLOG_DEBUG("Detected.");
  return targets_;
}

void ArmorDetector::VisualizeResult(const cv::Mat &output, int verbose) {
  if (verbose > 0) {
    cv::drawContours(output, contours_, -1, kRED);
    cv::drawContours(output, contours_poly_, -1, kYELLOW);
  }
  if (verbose > 1) {
    std::ostringstream buf;
    buf << lightbars_.size() << " bars in " << duration_bars_.count() << " ms.";

    int baseLine;
    int v_pos = 0;
    cv::Size text_size =
        cv::getTextSize(buf.str(), kCV_FONT, 1.0, 2, &baseLine);
    v_pos += static_cast<int>(1.3 * text_size.height);
    cv::putText(output, buf.str(), cv::Point(0, v_pos), kCV_FONT, 1.0, kGREEN);

    buf.str(std::string());
    buf << targets_.size() << " armors in " << duration_armors_.count()
        << " ms.";
    text_size = cv::getTextSize(buf.str(), kCV_FONT, 1.0, 2, &baseLine);
    v_pos += static_cast<int>(1.3 * text_size.height);
    cv::putText(output, buf.str(), cv::Point(0, v_pos), kCV_FONT, 1.0, kGREEN);
  }
  VisualizeLightBar(output, verbose > 2);
  VisualizeArmor(output, verbose > 2);
}

#include "armor_detector.hpp"

#include "opencv2/opencv.hpp"
#include "spdlog/spdlog.h"

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;

namespace {

const auto kCV_FONT = cv::FONT_HERSHEY_SIMPLEX;
const cv::Scalar kGREEN(0., 255., 0.);
const cv::Scalar kRED(0., 0., 255.);
const cv::Scalar kYELLOW(0., 255., 255.);

}  // namespace

void ArmorDetector::InitDefaultParams(const std::string &params_path) {
  cv::FileStorage fs(params_path,
                     cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);

  fs << "binary_th" << 220;
  fs << "se_erosion" << 5;
  fs << "ap_erosion" << 1.;

  fs << "contour_size_low_th" << 0;
  fs << "contour_area_low_th" << 0.00001;
  fs << "contour_area_high_th" << 0.001;
  fs << "bar_area_low_th" << 0.00001;
  fs << "bar_area_high_th" << 0.001;
  fs << "angle_high_th" << 60;
  fs << "aspect_ratio_low_th" << 2;
  fs << "aspect_ratio_high_th" << 6;

  fs << "angle_diff_th" << 0.2;
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

    params_.contour_size_low_th = static_cast<int>(fs["contour_size_low_th"]);
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
  const auto start = high_resolution_clock::now();
  lightbars_.clear();
  targets_.clear();

  frame_size_ = frame.size();
  const double frame_area = frame_size_.area();

  std::vector<cv::Mat> channels(3);
  cv::Mat result;
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

  if (params_.se_erosion >= 0.) {
    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(2 * params_.se_erosion + 1, 2 * params_.se_erosion + 1));
    cv::morphologyEx(result, result, cv::MORPH_OPEN, kernel);
  }

  cv::findContours(result, contours_, cv::RETR_LIST,
                   cv::CHAIN_APPROX_TC89_KCOS);

#if 0 /* 平滑轮廓应该有用，但是这里简化轮廓没用 */
  contours_poly_.resize(contours_.size());
  for (size_t k = 0; k < contours_.size(); ++k) {
    cv::approxPolyDP(cv::Mat(contours_[k]), contours_poly_[k],
                     params_.ap_erosion, true);
  }
#endif

  SPDLOG_DEBUG("Found contours: {}", contours_.size());

  /* 检查轮廓是否为灯条 */
  for (const auto &contour : contours_) {
    /* 通过轮廓大小先排除明显不是的 */
    if (contour.size() < params_.contour_size_low_th) continue;

    /* 只留下轮廓大小在一定比例内的 */
    const double c_area = cv::contourArea(contour) / frame_area;
    if (c_area < params_.contour_area_low_th) continue;
    if (c_area > params_.contour_area_high_th) continue;

    LightBar potential_bar(cv::minAreaRect(contour));

    /* 灯条倾斜角度不能太大 */
    if (std::abs(potential_bar.Angle()) > params_.angle_high_th) continue;

    /* 灯条在画面中的大小要满足条件 */
    const double bar_area = potential_bar.Area() / frame_area;
    if (bar_area < params_.bar_area_low_th) continue;
    if (bar_area > params_.bar_area_high_th) continue;

    /* 灯条的长宽比要满足条件 */
    const double aspect_ratio = potential_bar.AspectRatio();
    if (aspect_ratio < params_.aspect_ratio_low_th) continue;
    if (aspect_ratio > params_.aspect_ratio_high_th) continue;

    lightbars_.emplace_back(potential_bar);
  }

  /* 从左到右排列找到的灯条 */
  std::sort(lightbars_.begin(), lightbars_.end(),
            [](LightBar &bar1, LightBar &bar2) {
              return bar1.Center().x < bar2.Center().x;
            });

  /* 记录运行时间 */
  const auto stop = high_resolution_clock::now();
  duration_bars_ = duration_cast<std::chrono::milliseconds>(stop - start);

  SPDLOG_DEBUG("Found {} light bars in {} ms.", lightbars_.size(),
               duration_bars_.count());
}

void ArmorDetector::MatchLightBars() {
  const auto start = high_resolution_clock::now();
  for (auto iti = lightbars_.begin(); iti != lightbars_.end(); ++iti) {
    for (auto itj = iti + 1; itj != lightbars_.end(); ++itj) {
      /* 两灯条角度差异 */
      const double angle_diff =
          algo::RelativeDifference(iti->Angle(), itj->Angle());

      /* 灯条是否朝同一侧倾斜 */
      const bool same_side = (iti->Angle() * itj->Angle()) > 0;

      if (same_side) {
        if (angle_diff > params_.angle_diff_th) continue;
      } else {
        /* 两侧时限制更严格 */
        if (angle_diff > (params_.angle_diff_th / 2.)) continue;
      }

      /* 灯条长度差异 */
      const double length_diff =
          algo::RelativeDifference(iti->Length(), itj->Length());
      if (length_diff > params_.length_diff_th) continue;

      /* 灯条高度差异 */
      const double height_diff =
          algo::RelativeDifference(iti->Center().y, itj->Center().y);
      if (height_diff > (params_.height_diff_th * frame_size_.height)) continue;

      /* 灯条面积差异 */
      const double area_diff =
          algo::RelativeDifference(iti->Area(), itj->Area());
      if (area_diff > params_.area_diff_th) continue;

      /* 灯条中心距离 */
      const double center_dist = cv::norm(iti->Center() - itj->Center());
      const double l = (iti->Length() + itj->Length()) / 2.;
      if (center_dist < l * params_.center_dist_low_th) continue;
      if (center_dist > l * params_.center_dist_high_th) continue;

      auto armor = Armor(*iti, *itj);
      targets_.emplace_back(armor);
      break;
    }
  }
  const auto stop = high_resolution_clock::now();
  duration_armors_ = duration_cast<std::chrono::milliseconds>(stop - start);
  SPDLOG_DEBUG("Found {} armors in {} ms.", targets_.size(),
               duration_armors_.count());
}

void ArmorDetector::VisualizeLightBar(const cv::Mat &output, bool add_lable) {
  if (!lightbars_.empty()) {
    for (auto &bar : lightbars_) {
      auto vertices = bar.Vertices();
      auto num_vertices = vertices.size();
      for (std::size_t i = 0; i < num_vertices; ++i)
        cv::line(output, vertices[i], vertices[(i + 1) % num_vertices], kGREEN);

      cv::drawMarker(output, bar.Center(), kGREEN, cv::MARKER_CROSS);

      if (add_lable) {
        cv::putText(output,
                    cv::format("%.2f, %.2f", bar.Center().x, bar.Center().y),
                    vertices[1], kCV_FONT, 1.0, kGREEN);
      }
    }
  }
}

void ArmorDetector::VisualizeArmor(const cv::Mat &output, bool add_lable) {
  if (!targets_.empty()) {
    for (auto &armor : targets_) {
      auto vertices = armor.SurfaceVertices();
      auto num_vertices = vertices.size();
      for (std::size_t i = 0; i < num_vertices; ++i) {
        cv::line(output, vertices[i], vertices[(i + 1) % num_vertices], kGREEN);
      }
      cv::drawMarker(output, armor.SurfaceCenter(), kGREEN, cv::MARKER_DIAMOND);

      if (add_lable) {
        cv::putText(output,
                    cv::format("%.2f, %.2f", armor.SurfaceCenter().x,
                               armor.SurfaceCenter().y),
                    vertices[1], kCV_FONT, 1.0, kGREEN);
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
  MatchLightBars();
  SPDLOG_DEBUG("Detected.");
  return targets_;
}

void ArmorDetector::VisualizeResult(const cv::Mat &output, int verbose) {
  if (verbose > 0) {
    cv::drawContours(output, contours_, -1, kRED);
    cv::drawContours(output, contours_poly_, -1, kYELLOW);
  }
  if (verbose > 1) {
    int baseLine, v_pos = 0;

    std::string label = cv::format("%ld bars in %ld ms.", lightbars_.size(),
                                   duration_bars_.count());
    cv::Size text_size = cv::getTextSize(label, kCV_FONT, 1.0, 2, &baseLine);
    v_pos += static_cast<int>(1.3 * text_size.height);
    cv::putText(output, label, cv::Point(0, v_pos), kCV_FONT, 1.0, kGREEN);

    label = cv::format("%ld armors in %ld ms.", targets_.size(),
                       duration_armors_.count());
    text_size = cv::getTextSize(label, kCV_FONT, 1.0, 2, &baseLine);
    v_pos += static_cast<int>(1.3 * text_size.height);
    cv::putText(output, label, cv::Point(0, v_pos), kCV_FONT, 1.0, kGREEN);
  }
  VisualizeLightBar(output, verbose > 2);
  VisualizeArmor(output, verbose > 2);
}

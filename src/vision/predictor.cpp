#include "predictor.hpp"

#include <ctime>

#include "common.hpp"

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;

namespace {

const auto kCV_FONT = cv::FONT_HERSHEY_SIMPLEX;
const cv::Scalar kGREEN(0., 255., 0.);
const cv::Scalar kRED(0., 0., 255.);
const cv::Scalar kYELLOW(0., 255., 255.);
const std::chrono::seconds kGAME_TIME(150);

const double kDELTA = 3;  //总延迟时间

}  // namespace

static double Angle(const cv::Point2f &p, const cv::Point2f &ctr) {
  auto rel = p - ctr;
  return std::atan2(rel.x, rel.y);
}

static double DeltaTheta(double t) {
  return 1.305 * kDELTA +
         0.785 / 1.884 * (cos(1.884 * t) - cos(1.884 * (t + kDELTA)));
}

void Predictor::MatchDirection() {
  const auto start = std::chrono::system_clock::now();
  SPDLOG_WARN("start MatchDirection");

  if (direction_ == component::Direction::kUNKNOWN) {
    cv::Point2f center = buff_.GetCenter();
    double angle, sum = 0;
    std::vector<double> angles;

    if (circumference_.size() == 5) {
      for (auto point : circumference_) {
        angle = Angle(point, center);
        angles.emplace_back(angle);
      }

      for (auto i = circumference_.size(); i > circumference_.size() - 4; i--) {
        double delta = angles[i] - angles[i - 1];
        sum += delta;
      }

      if (sum > 0)
        direction_ = component::Direction::kCCW;
      else if (sum == 0)
        direction_ = component::Direction::kUNKNOWN;
      else
        direction_ = component::Direction::kCW;

      circumference_.emplace_back(buff_.GetTarget().ImageCenter());
      SPDLOG_DEBUG("Back of circumference point {}, {}.",
                   buff_.GetTarget().ImageCenter().x,
                   buff_.GetTarget().ImageCenter().y);
    }

    SPDLOG_WARN("Buff's Direction is {}",
                component::DirectionToString(GetDirection()));
    const auto stop = std::chrono::system_clock::now();
    duration_direction_ =
        duration_cast<std::chrono::milliseconds>(stop - start);
  }
}

Armor Predictor::RotateArmor(const Armor &armor, double theta,
                             const cv::Point2f &center) {
  cv::Point2f predict_point[4];
  cv::Matx22d rot(cos(theta), -sin(theta), sin(theta), cos(theta));

  auto vertices = armor.ImageVertices();
  for (int i = 0; i < 3; i++) {
    cv::Matx21d vec(vertices[i].x - center.x, vertices[i].y - center.y);
    cv::Matx21d mat = rot * vec;
    predict_point[i] =
        cv::Point2f(mat.val[0] + center.x, mat.val[1] + center.y);
  }
  return Armor(
      cv::RotatedRect(predict_point[0], predict_point[1], predict_point[2]));
}

void Predictor::MatchPredict() {
  const auto start = std::chrono::system_clock::now();
  SetPredict(Armor());

  if (cv::Point2f(0, 0) == buff_.GetCenter()) {
    SPDLOG_ERROR("Center is empty.");
    return;
  }
  if (cv::Point2f(0, 0) == buff_.GetTarget().ImageCenter()) {
    SPDLOG_ERROR("Target center is empty.");
    return;
  }
  if (component::Direction::kUNKNOWN == direction_) return;

  cv::Point2f target_center = buff_.GetTarget().ImageCenter();
  cv::Point2f center = buff_.GetCenter();
  SPDLOG_DEBUG("center is {},{}", buff_.GetCenter().x, buff_.GetCenter().y);
  component::Direction direction = GetDirection();
  Armor predict;

  double angle = Angle(target_center, center);
  double theta = DeltaTheta(GetTime());  // GetTime()
  SPDLOG_WARN("Delta theta : {}", theta);
  while (angle > 90) angle -= 90;
  if (direction == component::Direction::kCW) theta = -theta;

  theta = theta / 180 * CV_PI;
  SPDLOG_WARN("Theta : {}", theta);
  Armor armor = RotateArmor(buff_.GetTarget(), theta, center);
  SetPredict(armor);
  const auto stop = std::chrono::system_clock::now();
  duration_predict_ = duration_cast<std::chrono::milliseconds>(stop - start);
}

Predictor::Predictor() { SPDLOG_TRACE("Constructed."); }

Predictor::Predictor(const std::vector<Buff> &buffs) {
  if (circumference_.size() < 5)
    for (auto buff : buffs) {
      circumference_.push_back(buff.GetTarget().ImageCenter());
      SPDLOG_DEBUG("Get Buff Center{},{} ", buff_.GetTarget().ImageCenter().x,
                   buff_.GetTarget().ImageCenter().y);
    }
  buff_ = buffs.back();
  num_ = buff_.GetArmors().size();
  SPDLOG_TRACE("Constructed.");
}

Predictor::~Predictor() { SPDLOG_TRACE("Destructed."); }

const Buff &Predictor::GetBuff() const { return buff_; }

void Predictor::SetBuff(const Buff &buff) {
  SPDLOG_DEBUG("Buff center is {}, {}", buff.GetCenter().x, buff.GetCenter().y);
  buff_ = buff;
}

const Armor &Predictor::GetPredict() const { return predict_; }

void Predictor::SetPredict(const Armor &predict) {
  SPDLOG_DEBUG("Predict center is {},{}", predict.ImageCenter().x,
               predict.ImageCenter().y);
  predict_ = predict;
}

component::Direction Predictor::GetDirection() {
  SPDLOG_DEBUG("Direction : {}", component::DirectionToString(direction_));
  return direction_;
}

void Predictor::SetDirection(component::Direction direction) {
  direction_ = direction;
}

double Predictor::GetTime() const {
  auto time = end_time_ - high_resolution_clock::now();
  SPDLOG_WARN("time_: {}ms", time.count() / 1000000.);
  return (double)time.count() / 1000000.;
}

void Predictor::SetTime(double time) {
  double duration = 90 - time;
  SPDLOG_WARN("duration : {}", duration);
  auto now = high_resolution_clock::now();
  auto end = now + std::chrono::seconds((int64_t)duration);
  end_time_ = end;

  std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  SPDLOG_WARN("Now Ctime : {}", std::ctime(&now_time));
  SPDLOG_WARN("End Ctime : {}", std::ctime(&end_time));
}

void Predictor::ResetTime() {
  if (buff_.GetArmors().size() < num_) SetTime(0);
  SPDLOG_WARN("Reset time.");
}

std::vector<Armor> Predictor::Predict() {
  SPDLOG_DEBUG("Predicting.");
  MatchDirection();
  MatchPredict();
  SPDLOG_ERROR("Predicted.");
  std::vector<Armor> targets = {predict_};
  return targets;
}

void Predictor::VisualizePrediction(const cv::Mat &output, bool add_lable) {
  SPDLOG_DEBUG("{}, {}", predict_.ImageCenter().x, predict_.ImageCenter().y);
  if (cv::Point2f(0, 0) != predict_.ImageCenter()) {
    auto vertices = predict_.ImageVertices();
    for (std::size_t i = 0; i < vertices.size(); ++i)
      cv::line(output, vertices[i], vertices[(i + 1) % 4], kYELLOW, 8);
    cv::line(output, buff_.GetCenter(), predict_.ImageCenter(), kRED, 3);
    if (add_lable) {
      std::ostringstream buf;
      std::string label;
      int baseLine, v_pos = 0;

      buf << predict_.ImageCenter().x << ", " << predict_.ImageCenter().y;
      cv::putText(output, buf.str(), vertices[1], kCV_FONT, 1.0, kRED);

      label = cv::format("Direction %s in %ld ms.",
                         component::DirectionToString(direction_).c_str(),
                         duration_direction_.count());
      cv::Size text_size = cv::getTextSize(label, kCV_FONT, 1.0, 2, &baseLine);
      v_pos += 3 * static_cast<int>(1.3 * text_size.height);
      cv::putText(output, label, cv::Point(0, v_pos), kCV_FONT, 1.0, kGREEN);

      label = cv::format("Find predict in %ld ms.", duration_predict_.count());
      v_pos += static_cast<int>(1.3 * text_size.height);
      cv::putText(output, label, cv::Point(0, v_pos), kCV_FONT, 1.0, kGREEN);
    }
  }
}
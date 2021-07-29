#include "predictor.hpp"

#include <ctime>

using std::chrono::high_resolution_clock;

namespace {

const auto kCV_FONT = cv::FONT_HERSHEY_SIMPLEX;
const cv::Scalar kGREEN(0., 255., 0.);
const cv::Scalar kRED(0., 0., 255.);
const cv::Scalar kYELLOW(0., 255., 255.);
const std::chrono::seconds kGAME_TIME(150);

const double kDELTA = 0.3;  //总延迟时间

}  // namespace

static double Angle(const cv::Point2f &p, const cv::Point2f &ctr) {
  auto rel = p - ctr;
  return std::atan2(rel.x, rel.y);
}

// TODO: Flag你大爷，你给我好好写
static double Speed(double temp, bool flag) {
  if (flag)
    temp = 0.785 * std::sin(1.884 * temp) + 1.305;
  else
    temp = std::asin((temp - 1.305) / 1.884) / 0.785;
  return temp;
}

static double DeltaTheta(double t) {
  return 1.305 * kDELTA +
         0.785 / 1.884 * (cos(1.884 * t) - cos(1.884 * (t + kDELTA)));
}

void Predictor::MatchDirection() {
  SPDLOG_DEBUG("start MatchDirection");
  if (direction_ == common::Direction::kUNKNOWN) {
    cv::Point2f center = buff_.GetCenter();
    double angle, sum = 0;
    std::vector<double> angles;

    if (circumference_.size() == 5) {
      for (auto point : circumference_) {
        angle = Angle(point, center);
        angles.emplace_back(angle);
      }

      for (int i = 4; i > 1; i--) {
        double delta = angles[i] - angles[i - 1];
        sum += delta;
      }

      if (sum > 0)
        direction_ = common::Direction::kCCW;
      else if (sum == 0)
        direction_ = common::Direction::kUNKNOWN;
      else
        direction_ = common::Direction::kCW;
    }

    else {
      circumference_.emplace_back(buff_.GetTarget().SurfaceCenter());
      SPDLOG_DEBUG("Emplace_back point {},{}.",
                   buff_.GetTarget().SurfaceCenter().x,
                   buff_.GetTarget().SurfaceCenter().y);
    }

    SPDLOG_DEBUG("buff_'s getDirection is {}", GetDirection());
  }
}

Armor Predictor::RotateArmor(const Armor &armor, double theta,
                             const cv::Point2f &center) {
  cv::Point2f predict_point[4];
  cv::Matx22d rot(cos(theta), -sin(theta), sin(theta), cos(theta));

  auto vertices = armor.SurfaceVertices();
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
  SetPredict(Armor());
  if (cv::Point2f(0, 0) == buff_.GetCenter()) return;
  if (cv::Point2f(0, 0) == buff_.GetTarget().SurfaceCenter()) return;
  if (common::Direction::kUNKNOWN == direction_) return;

  cv::Point2f target_center = buff_.GetTarget().SurfaceCenter();
  cv::Point2f center = buff_.GetCenter();
  SPDLOG_DEBUG("center is {},{}", buff_.GetCenter().x, buff_.GetCenter().y);
  common::Direction direction = GetDirection();
  Armor predict;

  double angle = Angle(target_center, center);
  SPDLOG_DEBUG("GetTime()");
  double theta = DeltaTheta(GetTime());  // GetTime()
  while (angle > 90) angle -= 90;
  if (direction == common::Direction::kCW) theta = -theta;

  theta = theta / 180 * CV_PI;
  SPDLOG_WARN("Theta : {}", theta);
  Armor armor = RotateArmor(buff_.GetTarget(), theta, center);
  SetPredict(armor);
}

Predictor::Predictor() { SPDLOG_TRACE("Constructed."); }

Predictor::Predictor(const std::vector<Buff> &buffs) {
  if (circumference_.size() < 5)
    for (auto buff : buffs)
      circumference_.push_back(buff.GetTarget().SurfaceCenter());
  buff_ = buffs.back();
  num_ = buff_.GetArmors().size();
  SPDLOG_TRACE("Constructed.");
}

Predictor::~Predictor() { SPDLOG_TRACE("Destructed."); }

const Buff &Predictor::GetBuff() const { return buff_; }

void Predictor::SetBuff(const Buff &buff) {
  SPDLOG_DEBUG("Buff has {} armors.", buff.GetArmors().size());
  buff_ = buff;
}

const Armor &Predictor::GetPredict() const { return predict_; }

void Predictor::SetPredict(const Armor &predict) {
  SPDLOG_DEBUG("Predict center is {},{}", predict.SurfaceCenter().x,
               predict.SurfaceCenter().y);
  predict_ = predict;
}

common::Direction Predictor::GetDirection() {
  SPDLOG_DEBUG("Direction : {}", common::DirectionToString(direction_));
  return direction_;
}

void Predictor::SetDirection(common::Direction direction) {
  direction_ = direction;
}

double Predictor::GetTime() const {
  auto time = end_time_ - std::chrono::high_resolution_clock::now();
  SPDLOG_WARN("time_: {}ms", time.count());
  return (double)time.count();
}

void Predictor::SetTime(double time) {
  double du = 150 - time;
  SPDLOG_WARN("duration : {}", du);
  auto now = high_resolution_clock::now();
  auto end = now + std::chrono::seconds((int64_t)du);
  end_time_ = end;

  std::time_t a = std::chrono::system_clock::to_time_t(now);
  std::time_t b = std::chrono::system_clock::to_time_t(end);
  SPDLOG_WARN("Now Ctime : {}", std::ctime(&a));
  SPDLOG_WARN("End Ctime : {}", std::ctime(&b));
}

void Predictor::ResetTime() {
  if (buff_.GetArmors().size() < num_) SetTime(0);
}

Buff Predictor::Predict() {
  SPDLOG_DEBUG("Predicting.");
  MatchDirection();
  MatchPredict();
  SPDLOG_DEBUG("Predicted.");
  Buff target(buff_.GetCenter(), buff_.GetArmors(), GetPredict());
  return target;
}

void Predictor::VisualizePrediction(const cv::Mat &output, bool add_lable) {
  Armor predict = GetPredict();
  if (cv::Point2f(0, 0) != predict.SurfaceCenter()) {
    auto vertices = predict.SurfaceVertices();
    for (std::size_t i = 0; i < vertices.size(); ++i)
      cv::line(output, vertices[i], vertices[(i + 1) % 4], kYELLOW, 8);
    cv::line(output, buff_.GetCenter(), predict_.SurfaceCenter(), kRED, 3);
    if (add_lable) {
      std::ostringstream buf;
      buf << predict.SurfaceCenter().x << ", " << predict.SurfaceCenter().y;
      cv::putText(output, buf.str(), vertices[1], kCV_FONT, 1.0, kRED);
    }
  }
}
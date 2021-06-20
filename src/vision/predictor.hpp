#pragma once

#include <chrono>
#include <vector>

#include "buff.hpp"
#include "buff_detector.hpp"
#include "common.hpp"
#include "opencv2/opencv.hpp"

class Predictor {
 private:
  Buff buff_;
  Armor predict_;
  std::size_t num_;
  std::chrono::system_clock::time_point end_time_;
  std::vector<cv::Point2f> circumference_;
  common::Direction direction_ = common::Direction::kUNKNOWN;

  void MatchDirection();
  void MatchPredict();
  Armor RotateArmor(const Armor &armor, double theta,
                    const cv::Point2f &center);

 public:
  Predictor();
  Predictor(const std::vector<Buff> &buffs);
  ~Predictor();

  const Buff &GetBuff() const;
  void SetBuff(const Buff &buff);

  const Armor &GetPredict() const;
  void SetPredict(const Armor &predict);

  common::Direction GetDirection();
  void SetDirection(common::Direction direction);

  double GetTime() const;
  void SetTime(double time);
  void ResetTime();

  Buff Predict();

  void VisualizePrediction(const cv::Mat &output, bool add_lable);
};

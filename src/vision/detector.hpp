#pragma once

#include <chrono>
#include <vector>

#include "armor.hpp"
#include "armor_classifier.hpp"
#include "game.hpp"
#include "light_bar.hpp"

template <typename Target, typename Param>
class Detector {
 private:
  virtual void InitDefaultParams(const std::string &params_path) = 0;
  virtual bool PrepareParams(const std::string &params_path) = 0;

 public:
  std::vector<Target> targets_;
  Param params_;

  virtual void LoadParams(const std::string &params_path) = 0;

  virtual const std::vector<Target> &Detect(const cv::Mat &frame) = 0;
  virtual void VisualizeResult(const cv::Mat &output,
                               bool add_lable = true) = 0;
};

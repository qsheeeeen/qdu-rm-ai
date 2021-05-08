#pragma once

#include <chrono>
#include <vector>

#include "opencv2/opencv.hpp"
#include "spdlog/spdlog.h"
#include "tbb/concurrent_vector.h"

template <typename Target, typename Param>
class Detector {
 public:
  cv::Size frame_size_;
  tbb::concurrent_vector<Target> targets_;
  Param params_;

  virtual void InitDefaultParams(const std::string &path) = 0;
  virtual bool PrepareParams(const std::string &path) = 0;

  void LoadParams(const std::string &path) {
    if (!PrepareParams(path)) {
      InitDefaultParams(path);
      PrepareParams(path);
      SPDLOG_WARN("Can not find parasm file. Created and reloaded.");
    }
    SPDLOG_DEBUG("Params loaded.");
  }

  virtual const tbb::concurrent_vector<Target> &Detect(
      const cv::Mat &frame) = 0;
  virtual void VisualizeResult(const cv::Mat &output, int verbose = 1) = 0;
};

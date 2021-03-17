#include "buff_detector.hpp"

#include "gtest/gtest.h"
#include "opencv2/opencv.hpp"

TEST(TestVision, TestBuffDetector) {
  BuffDetector buff_detector("../../../runtime/RMUT2021_Buff.json",
                             game::Team::kBLUE);

  cv::Mat frame = cv::imread("../../../image/test_buff.png", cv::IMREAD_COLOR);
  ASSERT_FALSE(frame.empty()) << "Can not opening image.";

  std::vector<Buff> buffs = buff_detector.Detect(frame);
  buff_detector.VisualizeResult(frame, 2);

  cv::Mat result = frame.clone();
  cv::imwrite("../../../image/test_buff_result.jpg", result);
  SUCCEED();
}
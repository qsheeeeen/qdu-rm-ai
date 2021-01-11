#include "armor_detector.hpp"

#include "gtest/gtest.h"
#include "opencv2/opencv.hpp"

TEST(TestVision, TestArmorDetector) {
  ArmorDetector armor_detector("../../../tests/vision/params_test.json",
                               game::Team::kBLUE);

  cv::Mat img = imread("../../../image/test.jpg", cv::IMREAD_COLOR);
  ASSERT_FALSE(img.empty()) << "Can not opening image.";

  armor_detector.Detect(img);

  cv::Mat result = img.clone();
  armor_detector.VisualizeResult(result, true, false, true);
  cv::imwrite("../../../image/test_bars.jpg", result);

  result = img.clone();
  armor_detector.VisualizeResult(result, false, true, false);
  cv::imwrite("../../../image/test_armor.jpg", result);

  SUCCEED();
}
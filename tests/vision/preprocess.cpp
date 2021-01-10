#include "gtest/gtest.h"
#include "opencv2/opencv.hpp"

TEST(TestPreprocess, TestBlueOnly) {
  cv::Mat img = imread("../../../image/test.jpg", cv::IMREAD_COLOR);
  ASSERT_FALSE(img.empty()) << "Can not opening image.";

  cvtColor(img, img, cv::COLOR_BGR2GRAY);
  cv::imwrite("../../../image/test_blue.jpg", img);

  cv::Mat blue = imread("../../../image/test_blue.jpg", cv::IMREAD_GRAYSCALE);
  ASSERT_FALSE(blue.empty()) << "Can not opening image.";
}

TEST(TestPreprocess, TestBlueMinusRed) {
  cv::Mat img = imread("../../../image/test.jpg", cv::IMREAD_COLOR);
  ASSERT_FALSE(img.empty()) << "Can not opening image.";

  cv::Mat channels[3];
  cv::split(img, channels);

  cv::Mat result = channels[0] - channels[2];
  cv::imwrite("../../../image/test_b-r.jpg", result);

  cv::Mat blue = imread("../../../image/test_b-r.jpg", cv::IMREAD_GRAYSCALE);
  ASSERT_FALSE(blue.empty()) << "Can not opening image.";
}

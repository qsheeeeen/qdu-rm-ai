#include <iostream>

#include "gtest/gtest.h"
#include "opencv2/opencv.hpp"

TEST(TestVision, ExampleTest) { ASSERT_EQ(1, 1); }

TEST(TestVision, OpencvBasicIo) {
  cv::Mat img = imread("../../../image/test.jpg", cv::IMREAD_COLOR);
  ASSERT_FALSE(img.empty()) << "Can not opening image.";

  cvtColor(img, img, cv::COLOR_BGR2GRAY);
  cv::imwrite("../../../image/test_gray.jpg", img);

  cv::Mat gray = imread("../../../image/test_gray.jpg", cv::IMREAD_GRAYSCALE);
  ASSERT_FALSE(gray.empty()) << "Can not opening image.";
}
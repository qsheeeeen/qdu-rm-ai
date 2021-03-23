#include "gtest/gtest.h"
#include "opencv2/opencv.hpp"

TEST(TestPreprocess, TestBlueOnly) {
  cv::Mat img = imread("../../../image/test.jpg", cv::IMREAD_COLOR);
  ASSERT_FALSE(img.empty()) << "Can not opening image.";

  cvtColor(img, img, cv::COLOR_BGR2GRAY);
  cv::imwrite("../../../image/test_blue.png", img);

  cv::Mat blue = imread("../../../image/test_blue.png", cv::IMREAD_GRAYSCALE);
  ASSERT_FALSE(blue.empty()) << "Can not opening image.";
}

TEST(TestPreprocess, TestBlueMinusRed) {
  cv::Mat img = imread("../../../image/test.jpg", cv::IMREAD_COLOR);
  ASSERT_FALSE(img.empty()) << "Can not opening image.";

  cv::Mat channels[3];
  cv::split(img, channels);

  cv::Mat result = channels[0] - channels[2];
  cv::imwrite("../../../image/test_b-r.png", result);

  cv::Mat blue = imread("../../../image/test_b-r.png", cv::IMREAD_GRAYSCALE);
  ASSERT_FALSE(blue.empty()) << "Can not opening image.";
}

TEST(TestPreprocess, TestJsonRead) {
  cv::FileStorage fs("../../../runtime/MV-CA016-10UC-6mm.json",
                     cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);

  ASSERT_TRUE(fs.isOpened());

  ASSERT_FALSE(fs["cam_mat"].mat().empty());
  ASSERT_TRUE(fs["cam_mat"].isMap());
  ASSERT_FALSE(fs["distor_coff"].empty());
  ASSERT_TRUE(fs["distor_coff"].isMap());

  fs.release();
}

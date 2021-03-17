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

TEST(TestPreprocess, TestJsonWR) {
  cv::FileStorage fs;
  fs.open("../../../tests/vision/test_cam_cali.json",
          cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);

  cv::Mat m(2, 2, CV_8UC3, cv::Scalar(0, 0, 255));

  ASSERT_TRUE(fs.isOpened());

  fs.startWriteStruct("MV-CA016-10UC-6mm", cv::FileNode::MAP);
  fs.write("cam_mat", m);
  fs.write("distor_coff", m);
  fs.endWriteStruct();
  fs.release();

  fs.open("../../../tests/vision/test_cam_cali.json",
          cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);

  ASSERT_FALSE(fs["MV-CA016-10UC-6mm"].empty());
  ASSERT_TRUE(fs["MV-CA016-10UC-6mm"].isMap());
  ASSERT_FALSE(fs["MV-CA016-10UC-6mm"]["cam_mat"].empty());
  ASSERT_TRUE(fs["MV-CA016-10UC-6mm"]["cam_mat"].isMap());

  cv::Mat r = fs["MV-CA016-10UC-6mm"]["cam_mat"].mat();
  ASSERT_FALSE(r.empty());

  ASSERT_TRUE(cv::sum(m != r) == cv::Scalar(0, 0, 0));

  fs.release();
}

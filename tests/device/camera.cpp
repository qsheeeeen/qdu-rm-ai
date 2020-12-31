#include "camera.hpp"

#include <fstream>

#include "gtest/gtest.h"
#include "opencv2/opencv.hpp"

TEST(TestCamera, TestCapture) {
  Camera cam;
  ASSERT_TRUE(cam.Open(0) == 0) << "Can not open camera 0.";

  cv::Mat frame = cam.GetFrame();
  ASSERT_FALSE(frame.empty()) << "Can not get frame from camera.";

  cv::imwrite("../../../image/test_capture.jpg", frame);
  std::ifstream f("../../../image/test_capture.jpg");
  ASSERT_TRUE(f.good()) << "Can not write frame to file.";
  f.close();

  cv::Mat img = imread("../../../image/test_capture.jpg", cv::IMREAD_COLOR);
  ASSERT_FALSE(img.empty()) << "Can not opening image after wrote.";
}

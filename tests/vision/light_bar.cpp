#include "light_bar.hpp"

#include "gtest/gtest.h"
#include "opencv2/opencv.hpp"

const cv::Point2f center(1.f, 1.f);
const cv::Size2f size(2.f, 3.f);
const float angle = 5.f;

const cv::RotatedRect test_rect(center, size, angle);

TEST(TestVision, TestLighjtBar) {
  LighjtBar light_bar(test_rect);

  ASSERT_EQ(light_bar.Center(), center);
  ASSERT_FLOAT_EQ(light_bar.Angle(), angle);
  ASSERT_GE(light_bar.Length(), size.height);
  ASSERT_GE(light_bar.Length(), size.width);
}

TEST(TestVision, TestLighjtBarInit) {
  LighjtBar light_bar;

  ASSERT_FLOAT_EQ(light_bar.Center().x, 0.f); 
  ASSERT_FLOAT_EQ(light_bar.Center().y, 0.f); 

  light_bar.Init(test_rect);

  ASSERT_EQ(light_bar.Center(), center);
  ASSERT_FLOAT_EQ(light_bar.Angle(), angle);
  ASSERT_GE(light_bar.Length(), size.height);
  ASSERT_GE(light_bar.Length(), size.width);
}

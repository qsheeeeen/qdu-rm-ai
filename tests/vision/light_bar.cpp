#include "light_bar.hpp"

#include "gtest/gtest.h"
#include "opencv2/opencv.hpp"

const cv::Point2f center(1., 1.);
const cv::Size2f size(2., 3.);
const double angle = 5.;

const cv::RotatedRect test_rect(center, size, angle);

TEST(TestVision, TestLightBar) {
  LightBar light_bar(test_rect);

  ASSERT_EQ(light_bar.Center(), center);
  ASSERT_FLOAT_EQ(light_bar.Angle(), angle);
  ASSERT_GE(light_bar.Length(), size.height);
  ASSERT_GE(light_bar.Length(), size.width);
  ASSERT_FLOAT_EQ(light_bar.Area(), size.area());
  ASSERT_FLOAT_EQ(light_bar.AspectRatio(), (3. / 2.));

  std::vector<cv::Point2f> p1 = light_bar.Vertices();
  cv::Point2f p2[4];
  test_rect.points(p2);
  ASSERT_EQ(p1.size(), 4);
  for (size_t i = 0; i < p1.size(); ++i) ASSERT_EQ(p1[i], p2[i]);
}

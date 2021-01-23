#include "armor.hpp"

#include "gtest/gtest.h"
#include "light_bar.hpp"
#include "opencv2/opencv.hpp"

const cv::RotatedRect left_rect(cv::Point2f(1., 3.), cv::Size2f(1., 3.), 5.);

const cv::RotatedRect right_rect(cv::Point2f(3., 1.), cv::Size2f(1., 3.), 7.);

const LightBar left_bar(left_rect);
const LightBar right_bar(right_rect);

const game::Model model = game::Model::kHERO;

TEST(TestVision, TestArmor) {
  Armor armor(left_bar, right_bar);

  ASSERT_FLOAT_EQ(armor.Center2D().x, 2.);
  ASSERT_FLOAT_EQ(armor.Center2D().y, 2.);
  ASSERT_FLOAT_EQ(armor.Angle2D(), 6.);

  ASSERT_EQ(armor.GetModel(), game::Model::kUNKNOWN);

  armor.SetModel(model);
  ASSERT_EQ(armor.GetModel(), model);
}

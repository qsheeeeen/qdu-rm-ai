#include "armor.hpp"

#include "gtest/gtest.h"
#include "light_bar.hpp"
#include "opencv2/opencv.hpp"

const cv::RotatedRect left_rect(cv::Point2f(1.f, 3.f), cv::Size2f(1.f, 3.f),
                                5.f);

const cv::RotatedRect right_rect(cv::Point2f(3.f, 1.f), cv::Size2f(1.f, 3.f),
                                 7.f);

const LightBar left_bar(left_rect);
const LightBar right_bar(right_rect);

const game::Model model =  game::Model::kHERO;

TEST(TestVision, TestArmor) {
  Armor armor(left_bar, right_bar);

  ASSERT_FLOAT_EQ(armor.Center().x, 2.f);
  ASSERT_FLOAT_EQ(armor.Center().y, 2.f);
  ASSERT_FLOAT_EQ(armor.Angle(), 6.f);

  ASSERT_EQ(armor.GetModel(), game::Model::kUNKNOWN);

  armor.SetModel(model);
  ASSERT_EQ(armor.GetModel(), model);
}

TEST(TestVision, TestArmorInit) {
  Armor armor;

  armor.Init(left_bar, right_bar);

  ASSERT_FLOAT_EQ(armor.Center().x, 2.f);
  ASSERT_FLOAT_EQ(armor.Center().y, 2.f);
  ASSERT_FLOAT_EQ(armor.Angle(), 6.f);
  
  ASSERT_EQ(armor.GetModel(), game::Model::kUNKNOWN);

  armor.SetModel(model);
  ASSERT_EQ(armor.GetModel(), model);
}
#include "armor_classifier.hpp"

#include "armor.hpp"
#include "common.hpp"
#include "gtest/gtest.h"

TEST(TestVision, TestArmorClassifier) {
  ArmorClassifier armor_classifier("../../../runtime/armor_classifier.onnx", 28,
                                   28);
  cv::Mat f = cv::imread("../../../image/model_sticker/s2.png");
  Armor armor(cv::RotatedRect(cv::Point2f(0, 0), cv::Point2f(f.cols, 0),
                              cv::Point2f(f.cols, f.rows)));
  armor_classifier.ClassifyModel(armor, f);
  ASSERT_EQ(armor.GetModel(), game::Model::kINFANTRY);
}

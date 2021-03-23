#include "armor_classifier.hpp"

#include "armor.hpp"
#include "common.hpp"
#include "gtest/gtest.h"

TEST(TestVision, TestArmorClassifier) {
  ArmorClassifier armor_classifier(
      "../../../runtime/armor_classifier.onnx",
      "../../../runtime/armor_classifier_lable.json", cv::Size(28, 28));
  cv::Mat f = cv::imread("../../../image/p6.png");
  Armor armor(cv::RotatedRect(cv::Point2f(0, 0), cv::Point2f(f.cols, 0),
                              cv::Point2f(f.cols, f.rows)));

  cv::imwrite("../../../image/test_face.jpg", armor.Face(f));
  cv::Mat nn_input;
  cv::resize(armor.Face(f), nn_input, cv::Size(28, 28));
  cv::imwrite("../../../image/test_nn_input.jpg", nn_input);
  armor_classifier.ClassifyModel(armor, f);
  ASSERT_EQ(armor.GetModel(), game::Model::kHERO);
}

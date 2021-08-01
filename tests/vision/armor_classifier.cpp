#include "armor_classifier.hpp"

#include "armor.hpp"
#include "common.hpp"
#include "gtest/gtest.h"

ArmorClassifier armor_classifier("../../../runtime/armor_classifier.onnx",
                                 "../../../runtime/armor_classifier_lable.json",
                                 cv::Size(28, 28));

game::Model Classify(const std::string& path) {
  cv::Mat f = cv::imread(path);
  Armor armor(cv::RotatedRect(cv::Point2f(0, 0), cv::Point2f(f.cols, 0),
                              cv::Point2f(f.cols, f.rows)));
  armor_classifier.ClassifyModel(armor, f);
  return armor.GetModel();
}

TEST(TestVision, TestArmorClassifier) {
  ASSERT_EQ(Classify("../../../image/test_classifier_0.png"),
            game::Model::kINFANTRY);
  ASSERT_EQ(Classify("../../../image/test_classifier_1.png"),
            game::Model::kENGINEER);
  ASSERT_EQ(Classify("../../../image/test_classifier_2.png"),
            game::Model::kHERO);
  ASSERT_EQ(Classify("../../../image/test_classifier_3.png"),
            game::Model::kHERO);
}

TEST(TestVision, TestArmorClassifierInput) {
  cv::Mat f = cv::imread("../../../image/p2.png");
  Armor armor(cv::RotatedRect(cv::Point2f(0, 0), cv::Point2f(f.cols, 0),
                              cv::Point2f(f.cols, f.rows)));

  cv::imwrite("../../../image/test_face.png", armor.ImageFace(f));
  cv::Mat nn_input;
  cv::resize(armor.ImageFace(f), nn_input, cv::Size(28, 28));
  cv::imwrite("../../../image/test_nn_input.png", nn_input);
}

#include "armor_detector.hpp"

#include "gtest/gtest.h"
#include "opencv2/opencv.hpp"

TEST(TestVision, TestArmorDetector) {
  ArmorDetector armor_detector("../../../tests/vision/params_test.json",
                               game::Team::kBLUE);

  cv::Mat img = imread("../../../image/test.jpg", cv::IMREAD_COLOR);
  ASSERT_FALSE(img.empty()) << "Can not opening image.";

  std::vector<Armor> armors = armor_detector.Detect(img);
  EXPECT_EQ(armors.size(), 6) << "Can not detect armor in original image.";

  cv::Mat result = img.clone();
  armor_detector.VisualizeResult(result, 2);
  cv::imwrite("../../../image/test_origin.jpg", result);

  cv::resize(img, img, cv::Size(640, 426));

  armors = armor_detector.Detect(img);
  EXPECT_EQ(armors.size(), 6) << "Can not detect armor in small image.";

  result = img.clone();
  armor_detector.VisualizeResult(result, 1);
  cv::imwrite("../../../image/test_resized.jpg", result);

  armor_detector.SetEnemyTeam(game::Team::kRED);
  armors = armor_detector.Detect(img);
  EXPECT_EQ(armors.size(), 0) << "Can not tell the enemy from ourselves.";

  SUCCEED();
}
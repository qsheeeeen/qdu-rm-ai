#include "armor_detector.hpp"

#include "gtest/gtest.h"
#include "opencv2/opencv.hpp"

TEST(TestVision, TestArmorDetector) {
  ArmorDetector armor_detector("../../../runtime/test_params.json",
                               game::Team::kBLUE);

  cv::Mat img = imread("../../../image/test.jpg", cv::IMREAD_COLOR);
  ASSERT_FALSE(img.empty()) << "Can not opening image.";

  tbb::concurrent_vector<Armor> armors = armor_detector.Detect(img);
  EXPECT_EQ(armors.size(), 6) << "Can not detect armor in original image.";

  cv::Mat result = img.clone();
  armor_detector.VisualizeResult(result, 2);
  cv::imwrite("../../../image/test_origin.png", result);

  for (size_t i = 0; i < armors.size(); ++i) {
    cv::imwrite(cv::format("../../../image/p%ld.png", i), armors[i].Face(img));
  }

  cv::resize(img, img, cv::Size(640, 426));

  armors = armor_detector.Detect(img);
  EXPECT_EQ(armors.size(), 6) << "Can not detect armor in small image.";

  result = img.clone();
  armor_detector.VisualizeResult(result, 1);
  cv::imwrite("../../../image/test_resized.png", result);

  armor_detector.SetEnemyTeam(game::Team::kRED);
  armors = armor_detector.Detect(img);
  EXPECT_EQ(armors.size(), 0) << "Can not tell the enemy from ourselves.";
}
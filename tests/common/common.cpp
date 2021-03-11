#include "common.hpp"

#include "gtest/gtest.h"

TEST(TestVision, TestGame) {
  ASSERT_TRUE(game::HasBigArmor(game::Model::kHERO));
  ASSERT_TRUE(game::HasBigArmor(game::Model::kSENTRY));
}

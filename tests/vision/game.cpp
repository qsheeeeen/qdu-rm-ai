#include "game.hpp"

#include "gtest/gtest.h"

TEST(TestVision, TestGame) {
  ASSERT_TRUE(game::HasBigArmor(game::Model::kHERO));
  ASSERT_TRUE(game::HasBigArmor(game::Model::kSENTRY));
  ASSERT_EQ(1, 1);
}

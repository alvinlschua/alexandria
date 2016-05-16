#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundef"
#pragma clang diagnostic ignored "-Wdeprecated"
#pragma clang diagnostic ignored "-Wmissing-noreturn"
#pragma clang diagnostic ignored "-Wshift-sign-overflow"
#pragma clang diagnostic ignored "-Wused-but-marked-unused"
#include "gtest/gtest.h"
#pragma clang diagnostic pop

#include <vector>

#include "util/function_cache.h"

TEST(FunctionCache, Cache) {
  using Alexandria::FunctionCache;

  FunctionCache<int, std::vector<int>> fn(
      [](int value) { return std::vector<int>(10, value); });

  auto first = fn(1);
  auto second = fn(2);
  auto third = fn(3);

  EXPECT_TRUE(fn.isCached(1));
  EXPECT_TRUE(fn.isCached(2));
  EXPECT_TRUE(fn.isCached(3));

  auto forth = fn(2);

  EXPECT_EQ(second, forth);
  EXPECT_EQ(fn.cacheSize(), 3);

  fn.clearCache();
  EXPECT_EQ(fn.cacheSize(), 0);
}

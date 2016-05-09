#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundef"
#pragma clang diagnostic ignored "-Wdeprecated"
#pragma clang diagnostic ignored "-Wmissing-noreturn"
#pragma clang diagnostic ignored "-Wshift-sign-overflow"
#pragma clang diagnostic ignored "-Wused-but-marked-unused"
#include "gtest/gtest.h"
#pragma clang diagnostic pop

#include <vector>
#include "util/util.h"

TEST(Util, Gather) {
  using namespace std;
  using namespace Util;

  std::vector<int> x({1, 2, 3, 4});
  std::vector<int> indices({0, 0, 1, 3});
  std::vector<int> result(4);

  // result[i] = x[index[i]]
  gather(indices.cbegin(), indices.cend(), x.cbegin(), result.begin());
  EXPECT_EQ(result, std::vector<int>({1, 1, 2, 4}));

  std::vector<size_t> indices2({1, 0, 3, 2});
  gather(indices2.cbegin(), indices2.cend(), x.cbegin(), result.begin());
  EXPECT_EQ(result, std::vector<int>({2, 1, 4, 3}));
}

TEST(Util, Scatter) {
  using namespace std;
  using namespace Util;

  std::vector<double> x({1, 2, 3, 4});
  std::vector<size_t> indices({0, 0, 1, 3});
  std::vector<double> result(4, 0);

  // result[index[i]] = x[i]
  scatter(indices.cbegin(), indices.cend(), x.cbegin(), result.begin());
  EXPECT_EQ(result, std::vector<double>({2, 3, 0, 4}));

  std::vector<size_t> indices2({1, 0, 3, 2});
  scatter(indices2.cbegin(), indices2.cend(), x.cbegin(), result.begin());
  EXPECT_EQ(result, std::vector<double>({2, 1, 4, 3}));

  std::vector<size_t> indices3({2, 0, 1, 3});
  scatter(indices3.cbegin(), indices3.cend(), x.cbegin(), result.begin());
  EXPECT_EQ(result, std::vector<double>({2, 3, 1, 4}));
}
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundef"
#pragma clang diagnostic ignored "-Wdeprecated"
#pragma clang diagnostic ignored "-Wmissing-noreturn"
#pragma clang diagnostic ignored "-Wshift-sign-overflow"
#pragma clang diagnostic ignored "-Wused-but-marked-unused"
#include "gtest/gtest.h"
#pragma clang diagnostic pop

#include <sstream>

#include "tensor/addresser.h"

TEST(Addresser, Basic1) {
  using namespace Util;
  using namespace Tensor;
  using namespace std;

  Shape shape({2, 1, 4});
  Addresser addresser(&shape);

  EXPECT_EQ(addresser.flatIndex({0, 0, 0}), 0ul);
  EXPECT_EQ(addresser.flatIndex({0, 0, 1}), 1ul);
  EXPECT_EQ(addresser.flatIndex({0, 0, 2}), 2ul);
  EXPECT_EQ(addresser.flatIndex({0, 0, 3}), 3ul);
  EXPECT_EQ(addresser.flatIndex({1, 0, 0}), 4ul);
  EXPECT_EQ(addresser.flatIndex({1, 0, 1}), 5ul);
  EXPECT_EQ(addresser.flatIndex({1, 0, 2}), 6ul);
  EXPECT_EQ(addresser.flatIndex({1, 0, 3}), 7ul);

  EXPECT_EQ(addresser.address(0), Address({0, 0, 0}));
  EXPECT_EQ(addresser.address(1), Address({0, 0, 1}));
  EXPECT_EQ(addresser.address(2), Address({0, 0, 2}));
  EXPECT_EQ(addresser.address(3), Address({0, 0, 3}));
  EXPECT_EQ(addresser.address(4), Address({1, 0, 0}));
  EXPECT_EQ(addresser.address(5), Address({1, 0, 1}));
  EXPECT_EQ(addresser.address(6), Address({1, 0, 2}));
  EXPECT_EQ(addresser.address(7), Address({1, 0, 3}));

  auto shape2 = Shape({1, 2, 4});
  auto addresser2 = Addresser(&shape2);
  EXPECT_EQ(addresser2.flatIndex({0, 0, 0}), 0ul);
  EXPECT_EQ(addresser2.flatIndex({0, 0, 1}), 1ul);
  EXPECT_EQ(addresser2.flatIndex({0, 0, 2}), 2ul);
  EXPECT_EQ(addresser2.flatIndex({0, 0, 3}), 3ul);
  EXPECT_EQ(addresser2.flatIndex({0, 1, 0}), 4ul);
  EXPECT_EQ(addresser2.flatIndex({0, 1, 1}), 5ul);
  EXPECT_EQ(addresser2.flatIndex({0, 1, 2}), 6ul);
  EXPECT_EQ(addresser2.flatIndex({0, 1, 3}), 7ul);

  EXPECT_EQ(addresser2.address(0), Address({0, 0, 0}));
  EXPECT_EQ(addresser2.address(1), Address({0, 0, 1}));
  EXPECT_EQ(addresser2.address(2), Address({0, 0, 2}));
  EXPECT_EQ(addresser2.address(3), Address({0, 0, 3}));
  EXPECT_EQ(addresser2.address(4), Address({0, 1, 0}));
  EXPECT_EQ(addresser2.address(5), Address({0, 1, 1}));
  EXPECT_EQ(addresser2.address(6), Address({0, 1, 2}));
  EXPECT_EQ(addresser2.address(7), Address({0, 1, 3}));
}

TEST(Addresser, increment) {
  using namespace Tensor;
  using namespace std;

  Shape shape({3, 1, 2});
  Addresser addresser(&shape);

  Address address(3, 0);

  EXPECT_EQ(address, std::vector<size_t>({0, 0, 0}));

  address = addresser.increment(std::move(address));
  EXPECT_EQ(address, std::vector<size_t>({0, 0, 1}));

  address = addresser.increment(std::move(address));
  EXPECT_EQ(address, std::vector<size_t>({1, 0, 0}));

  address = addresser.increment(std::move(address));
  EXPECT_EQ(address, std::vector<size_t>({1, 0, 1}));

  address = addresser.increment(std::move(address));
  EXPECT_EQ(address, std::vector<size_t>({2, 0, 0}));

  address = addresser.increment(std::move(address));
  EXPECT_EQ(address, std::vector<size_t>({2, 0, 1}));

  address = Address(3, 0);
  auto step = 2ul;
  address = addresser.increment(std::move(address), step);
  EXPECT_EQ(address, std::vector<size_t>({1, 0, 0}));

  address = addresser.increment(std::move(address), step);
  EXPECT_EQ(address, std::vector<size_t>({2, 0, 0}));
}

int main(int argc, char** argv) {
  // Disables elapsed time by default.
  ::testing::GTEST_FLAG(print_time) = false;

  // This allows the user to override the flag on the command line.
  ::testing::InitGoogleTest(&argc, argv);

  google::InstallFailureSignalHandler();

  return RUN_ALL_TESTS();
}

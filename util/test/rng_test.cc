#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundef"
#pragma clang diagnostic ignored "-Wdeprecated"
#pragma clang diagnostic ignored "-Wmissing-noreturn"
#pragma clang diagnostic ignored "-Wshift-sign-overflow"
#pragma clang diagnostic ignored "-Wused-but-marked-unused"
#include "gtest/gtest.h"
#pragma clang diagnostic pop

#include <sstream>

#include "util/rng.h"

TEST(Rng, Serialize) {
  using namespace std;
  using namespace Util;

  // do this to start form some intermediate state
  rng().generate(uniform_int_distribution<int>(-2, 4), 100);

  ostringstream sout;
  ArchiveOut ar_out(&sout);
  ar_out % rng();

  auto result = rng().generate(uniform_int_distribution<int>(-2, 4), 5);

  istringstream sin(sout.str());
  ArchiveIn ar_in(&sin);
  ar_in % rng();

  auto result_restore = rng().generate(uniform_int_distribution<int>(-2, 4), 5);

  EXPECT_EQ(result, result_restore);
}

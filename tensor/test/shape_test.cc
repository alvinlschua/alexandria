#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundef"
#pragma clang diagnostic ignored "-Wdeprecated"
#pragma clang diagnostic ignored "-Wmissing-noreturn"
#pragma clang diagnostic ignored "-Wshift-sign-overflow"
#pragma clang diagnostic ignored "-Wused-but-marked-unused"
#include "gtest/gtest.h"
#pragma clang diagnostic pop

#include <sstream>

#include "tensor/shape.h"

TEST(Shape, Basic) {
  using namespace Alexandria;
  using namespace std;

  Shape shape({2, 1, 1, 2});
  EXPECT_EQ(shape.nDimensions(), 4ul);
  EXPECT_EQ(shape[0], 2ul);
  EXPECT_EQ(shape[1], 1ul);
  EXPECT_EQ(shape[2], 1ul);
  EXPECT_EQ(shape[3], 2ul);

  auto shape2 = shape;
  EXPECT_EQ(shape2, shape);
  EXPECT_EQ(nElements(shape), 4ul);

  EXPECT_THROW(Shape({3, 0, 2}), std::invalid_argument);
}

TEST(Shape, Serialize) {
  using namespace Alexandria;
  using namespace std;

  ostringstream sout;
  ArchiveOut ar_out(&sout);
  Shape shape_out({2, 3, 1, 5});
  ar_out % shape_out;

  istringstream sin(sout.str());
  ArchiveIn ar_in(&sin);
  Shape shape_in({2, 1});
  //Shape shape_in;
  //({2, 1});

  ar_in % shape_in;

  EXPECT_EQ(shape_out, shape_in);
}

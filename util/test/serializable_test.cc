#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundef"
#pragma clang diagnostic ignored "-Wdeprecated"
#pragma clang diagnostic ignored "-Wmissing-noreturn"
#pragma clang diagnostic ignored "-Wshift-sign-overflow"
#pragma clang diagnostic ignored "-Wused-but-marked-unused"
#include "gtest/gtest.h"
#pragma clang diagnostic pop

#include <sstream>

#include "util/serializable.h"

class TestClass : public Alexandria::Serializable {
 public:
  TestClass() : a_(0), b_('\0') {}
  TestClass(int a, char b) : a_(a), b_(b) {}

  int a() const { return a_; }
  char b() const { return b_; }

 private:
  void serializeInImpl(Alexandria::ArchiveIn& ar, size_t /*version*/) final {
    ar % a_ % b_;
  }
  void serializeOutImpl(Alexandria::ArchiveOut& ar) const final { ar % a_ % b_; }
  size_t serializeOutVersionImpl() const { return 0; }

  int a_;
  char b_;
};

TEST(Serializable, Class) {
  using std::istringstream;
  using std::ostringstream;
  using Alexandria::ArchiveIn;
  using Alexandria::ArchiveOut;

  // do this to start form some intermediate state
  TestClass class_out(20, 'd');

  ostringstream sout;
  ArchiveOut ar_out(&sout);
  ar_out % class_out;

  istringstream sin(sout.str());
  ArchiveIn ar_in(&sin);
  TestClass class_in;
  ar_in % class_in;

  EXPECT_EQ(class_out.a(), class_in.a());
  EXPECT_EQ(class_out.b(), class_in.b());
}

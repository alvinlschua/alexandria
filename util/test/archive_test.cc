#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundef"
#pragma clang diagnostic ignored "-Wdeprecated"
#pragma clang diagnostic ignored "-Wmissing-noreturn"
#pragma clang diagnostic ignored "-Wshift-sign-overflow"
#pragma clang diagnostic ignored "-Wused-but-marked-unused"
#include "gtest/gtest.h"
#pragma clang diagnostic pop

#include <sstream>

#include "util/archive_in.h"
#include "util/archive_out.h"
#include "util/util.h"

TEST(Archive, Primitive) {
  using std::ostringstream;
  using std::istringstream;
  using std::string;
  using Util::ArchiveOut;
  using Util::ArchiveIn;

  ostringstream sout;
  ArchiveOut ar_out(&sout);

  bool bool_out = false;
  ar_out % bool_out;

  char char_out = 'd';
  ar_out % char_out;

  signed char signed_char_out = 'Q';
  ar_out % signed_char_out;

  unsigned char unsigned_char_out = 'g';
  ar_out % unsigned_char_out;

  short int short_int_out = 10;
  ar_out % short_int_out;

  unsigned short int unsigned_short_int_out = 20;
  ar_out % unsigned_short_int_out;

  int int_out = 11;
  ar_out % int_out;

  unsigned int unsigned_int_out = 2001;
  ar_out % unsigned_int_out;

  long int long_int_out = -12;
  ar_out % long_int_out;

  unsigned long int unsigned_long_int_out = 3042;
  ar_out % unsigned_long_int_out;

  float float_out = 1.4f;
  ar_out % float_out;
  ;

  double double_out = 31.4;
  ar_out % double_out;

  long double long_double_out = 314.42;
  ar_out % long_double_out;

  string string_out = "abcdefg";
  ar_out % string_out;

  std::istringstream sin(sout.str());
  ArchiveIn ar_in(&sin);

  bool bool_in = true;
  ar_in % bool_in;
  EXPECT_EQ(bool_out, bool_in);

  char char_in = '\0';
  ar_in % char_in;
  EXPECT_EQ(char_out, char_in);

  signed char signed_char_in = '\0';
  ar_in % signed_char_in;
  EXPECT_EQ(signed_char_out, signed_char_in);

  unsigned char unsigned_char_in = '\0';
  ar_in % unsigned_char_in;
  EXPECT_EQ(unsigned_char_out, unsigned_char_in);

  short int short_int_in = 0;
  ar_in % short_int_in;
  EXPECT_EQ(short_int_out, short_int_in);

  unsigned short int unsigned_short_int_in = 0;
  ar_in % unsigned_short_int_in;
  EXPECT_EQ(unsigned_short_int_out, unsigned_short_int_in);

  int int_in = 0;
  ar_in % int_in;
  EXPECT_EQ(int_out, int_in);

  unsigned int unsigned_int_in = 0;
  ar_in % unsigned_int_in;
  EXPECT_EQ(unsigned_int_out, unsigned_int_in);

  long int long_int_in = 0;
  ar_in % long_int_in;
  EXPECT_EQ(long_int_out, long_int_in);

  unsigned long int unsigned_long_int_in = 0;
  ar_in % unsigned_long_int_in;
  EXPECT_EQ(unsigned_long_int_out, unsigned_long_int_in);

  float float_in = 0;
  ar_in % float_in;
  EXPECT_EQ(float_out, float_in);

  double double_in = 0;
  ar_in % double_in;
  EXPECT_EQ(double_out, double_in);

  long double long_double_in = 0;
  ar_in % long_double_in;
  EXPECT_EQ(long_double_out, long_double_in);

  string string_in;
  ar_in % string_in;
  EXPECT_EQ(string_out, string_in);
}

TEST(Archive, Array) {
  using std::ostringstream;
  using std::istringstream;
  using Util::ArchiveOut;
  using Util::ArchiveIn;

  ostringstream sout;
  ArchiveOut ar_out(&sout);
  std::array<int, 4> array_out({{4, 5, 7, 5}});
  ar_out % array_out;

  istringstream sin(sout.str());
  ArchiveIn ar_in(&sin);
  std::array<int, 4> array_in;

  ar_in % array_in;

  EXPECT_EQ(array_in, array_out);
}

TEST(Archive, Vector) {
  using std::ostringstream;
  using std::istringstream;
  using Util::ArchiveOut;
  using Util::ArchiveIn;

  ostringstream sout;
  ArchiveOut ar_out(&sout);
  std::vector<int> vec_out({4, 5, 7, 5});
  ar_out % vec_out;

  istringstream sin(sout.str());
  ArchiveIn ar_in(&sin);
  std::vector<int> vec_in;

  ar_in % vec_in;

  EXPECT_EQ(vec_in, vec_out);

  sin = istringstream(sout.str());
  std::vector<int> vec_in2({1, 2, 1});

  ar_in % vec_in2;

  EXPECT_EQ(vec_in2, vec_out);
}

TEST(Archive, UnorderedSet) {
  using std::ostringstream;
  using std::istringstream;
  using Util::ArchiveOut;
  using Util::ArchiveIn;

  ostringstream sout;
  ArchiveOut ar_out(&sout);
  std::unordered_set<int> set_out({4, 5, 7, 5});
  ar_out % set_out;

  istringstream sin(sout.str());
  ArchiveIn ar_in(&sin);
  std::unordered_set<int> set_in;

  ar_in % set_in;

  EXPECT_EQ(set_in, set_out);
}

#ifndef UTIL_UTIL_H_
#define UTIL_UTIL_H_

#include <algorithm>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-conversion"
#include "glog/logging.h"
#pragma clang diagnostic pop

namespace Alexandria {

inline uint64_t hashCombine(uint64_t seed, uint64_t hash_value) {
  return seed ^ (hash_value + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

template <typename TIterator>
uint64_t hash64(TIterator begin, TIterator end) {
  using Value = typename TIterator::value_type;
  std::hash<Value> hasher;
  return std::accumulate(begin + 1, end, hasher(*begin),
                         [&hasher](Value init, Value val) {
                           return hashCombine(init, hasher(val));
                         });
}

template <typename TA, typename TB>
inline bool almostEqual(TA A, TB B, int maxUlps = 4) {
  return almostEqual(static_cast<float>(A), static_cast<float>(B), maxUlps);
}

template <>
inline bool almostEqual(int A, int B, int /*maxUlps*/) {
  return A == B;
}

template <>
inline bool almostEqual(long A, long B, int /*maxUlps*/) {
  return A == B;
}

union floatAsInt {
  float val;
  int intVal;
};

template <>
inline bool almostEqual(float A, float B, int maxUlps) {
  // Make sure maxUlps is non-negative and small enough that the
  // default NAN won't compare as equal to anything.
  CHECK_GT(maxUlps, 0);
  CHECK_LT(maxUlps, 4 * 1024 * 1024);
  floatAsInt floatInt;

  floatInt.val = A;
  int aInt = floatInt.intVal;
  // Make aInt lexicographically ordered as a twos-complement int
  if (aInt < 0) aInt = 0x80000000l - aInt;
  // Make bInt lexicographically ordered as a twos-complement int

  floatInt.val = B;
  int bInt = floatInt.intVal;
  if (bInt < 0) bInt = 0x80000000l - bInt;
  int intDiff = abs(aInt - bInt);
  if (intDiff <= maxUlps) return true;
  return false;
}

// Compute sgn of a value.
template <typename T>
int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

// TODO(alschua) may need to move this to a cc file
static int invalid_index = std::numeric_limits<int>::max();

// The gather operation computes
//    result[i] = index[i] != invalid_index ? vector[index[i]] : result[i]
// over all i.
template <typename TIterator, typename IndexIterator, typename ResultIterator>
inline void gather(IndexIterator index_begin, IndexIterator index_end,
                   TIterator begin, ResultIterator result_begin,
                   std::function<int(int)> reindex = [](int index) {
                     return index;
                   }) {
  auto iter = index_begin;
  auto riter = result_begin;
  for (; iter != index_end; ++iter, ++riter) {
    auto new_index = reindex(*iter);
    if (new_index == invalid_index) continue;
    (*riter) = *(begin + new_index);
  }
}

// The scatter operation computes
//   if(index[i] != invalid_index)
//     result[index[i]] = vector[i]
// over all i.
template <typename TIterator, typename IndexIterator, typename ResultIterator>
inline void scatter(IndexIterator index_begin, IndexIterator index_end,
                    TIterator begin, ResultIterator result_begin,
                    std::function<int(int)> reindex = [](int index) {
                      return index;
                    }) {
  typename IndexIterator::difference_type count = 0;
  std::for_each(index_begin, index_end,
                [result_begin, begin, &count, &reindex](int old_index) {
                  auto index = reindex(old_index);
                  if (index != invalid_index) {
                    *(result_begin + index) = *(begin + count);
                  }
                  count++;
                });
}

class unimplemented_exception : public std::exception {
 public:
  unimplemented_exception(const std::string& msg) : msg_(msg) {}
  unimplemented_exception(const unimplemented_exception&) = default;
  virtual ~unimplemented_exception() {}
  const char* what() const noexcept final { return msg_.c_str(); }

 private:
  std::string msg_;
};

}  // namespace Alexandria

#endif  // UTIL_UTIL_H_

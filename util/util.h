#ifndef UTIL_UTIL_H
#define UTIL_UTIL_H

#include <functional>
#include <iterator>
#include <numeric>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-conversion"
#include "glog/logging.h"
#pragma clang diagnostic pop

namespace Util {

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

template <typename T>
int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

// The gather operation computes result[i] = vector[index[i]] over all i.
template <typename TIterator, typename IndexIterator, typename ResultIterator>
inline void gather(IndexIterator index_begin, IndexIterator index_end,
                   TIterator begin, ResultIterator result_begin) {
  std::transform(
      index_begin, index_end, result_begin,
      [begin](typename IndexIterator::value_type index) {
        return *(begin +
                 static_cast<typename IndexIterator::difference_type>(index));
      });
}

// The scatter operation computes result[index[i]] = vector[i] over all i.
template <typename TIterator, typename IndexIterator, typename ResultIterator>
inline void scatter(IndexIterator index_begin, IndexIterator index_end,
                    TIterator begin, ResultIterator result_begin) {
  typename IndexIterator::difference_type count = 0;
  std::for_each(
      index_begin, index_end,
      [result_begin, begin, &count](typename IndexIterator::value_type index) {
        *(result_begin + static_cast<typename IndexIterator::difference_type>(
                             index)) = *(begin + count++);
      });
}

}  // namespace Util

#endif

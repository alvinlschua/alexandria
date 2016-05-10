#ifndef UTIL_RNG_H_
#define UTIL_RNG_H_

#include <algorithm>
#include <random>
#include <mutex>
#include <vector>

#include "util/singleton.h"
#include "util/serializable.h"

namespace Util {

// Random number generator singleton.
class Rng : public Singleton<Rng>, public Serializable {
 public:
  virtual ~Rng() {}

  template <typename TDistribution>
  using result_type = std::vector<typename TDistribution::result_type>;

  template <typename TDistribution>
  result_type<TDistribution> generate(TDistribution distribution, size_t n = 1);

  /* TODO(alvin): add function to set seed */

 private:
  void serializeInImpl(ArchiveIn& ar, size_t version) final;
  void serializeOutImpl(ArchiveOut& ar) const final;
  size_t serializeOutVersionImpl() const final { return 0; }

  mutable std::mutex mutex_;
  std::mt19937 generator_;
};

template <typename TDistribution>
Rng::result_type<TDistribution> Rng::generate(TDistribution distribution,
                                              size_t n) {
  std::lock_guard<std::mutex> lock(mutex_);
  result_type<TDistribution> result;

  result.reserve(n);
  generate_n(back_inserter(result), n,
             [this, &distribution]() { return distribution(generator_); });

  return result;
}

inline Rng& rng() { return Rng::instance(); }

}  // namespace Util

#endif  // UTIL_RNG_H_

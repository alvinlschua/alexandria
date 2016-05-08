#ifndef UTIL_FUNCTION_CACHE_H
#define UTIL_FUNCTION_CACHE_H

#include <memory>
#include <unordered_map>

#include "glog/logging.h"

namespace Util {

// A function cache to recall results of functions with expansive calculations.
// The input must have a hash.
template <typename TInput, typename TResult, typename THash = std::hash<TInput>>
class FunctionCache {
 public:
  using Input = TInput;
  using Result = TResult;
  using Hash = THash;

  FunctionCache(std::function<Result(Input)> function) : function_(function) {}

  // Is the result of the input cached?
  bool isCached(const Input& input) const;

  // Get that value from the cache or calculate and cache it.
  const Result& operator()(const Input& input);

  // Clear the cache.
  void clearCache() { cache_.clear(); }

  // Return the size of the cache.
  size_t cacheSize() const { return cache_.size(); }

 private:
  std::function<Result(Input)> function_;
  std::unordered_map<Input, Result, Hash> cache_;
};

template <typename TInput, typename TResult, typename THash>
bool FunctionCache<TInput, TResult, THash>::isCached(const Input& input) const {
  return cache_.find(input) != cache_.end();
}

template <typename TInput, typename TResult, typename THash>
auto FunctionCache<TInput, TResult, THash>::operator()(const Input& input)
    -> const Result & {
  auto iter = cache_.find(input);
  if (iter == cache_.end()) {
    auto result = false;
    std::tie(iter, result) = cache_.emplace(input, function_(input));
    CHECK(result) << "Cannot insert into cache";
  }

  return iter->second;
}

}  // namespace Util

#endif

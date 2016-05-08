#ifndef UTIL_CLOANABLE_H
#define UTIL_CLOANABLE_H

#include <memory>

namespace Util {

// Clonable interface.
//
// Requires concrete instances of the base class to be clonable.
// Returns the clone of an object as a unique pointer of the base class type.
template <typename T>
class Clonable {
 public:
  virtual ~Clonable() {}
  Clonable() = default;
  Clonable(const Clonable&) = default;
  Clonable& operator=(const Clonable&) = default;

  std::unique_ptr<T> clone() const { return cloneImpl(); }

 private:
  virtual std::unique_ptr<T> cloneImpl() const = 0;
};

}  // namespace Util

#endif

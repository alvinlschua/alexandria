#ifndef TENSOR_TENSOR_BASE_H
#define TENSOR_TENSOR_BASE_H

#include <vector>

#include "tensor/shape.h"
#include "util/clonable.h"
#include "util/serializable.h"

namespace Alexandria {

// This is a general tensor class.  As far as possible, transformations are
// done in-place eagerly.
//
// We also do not provide natural operators to have a more consistent interface.
// This results in a more verbose and less natural looking instruction set but
// encourages the use of less variables and avoid unnecessary copying.
//
// Some operations have restrictions on the way data is accessed. This is to
// make sure the class is used efficiently.
template <typename T>
class Tensor<T>::Base : public Serializable, public Clonable<Tensor<T>::Base> {
 public:
  virtual ~Base() {}

  Base() {}
  Base(const Base&) = default;
  Base& operator=(const Base&) = default;

  // Return the number of elements.
  size_t size() const { return sizeImpl(); }

  // Return the shape.
  const Shape& shape() const { return shapeImpl(); }

  // Access a const element.
  T at(const Address& address) const { return atImpl(address); }

  // Access a const element.
  T operator[](const Address& address) const { return atImpl(address); }

  // Set a value at the address.
  void set(const Address& address, T value, std::function<T(T, T)> fn) {
    setImpl(address, value, fn);
  }

  // Address iterators.
  AddressIterator begin() const { return beginImpl(); }
  AddressIterator end() const { return endImpl(); }

 private:
  virtual size_t sizeImpl() const = 0;
  virtual const Shape& shapeImpl() const = 0;
  virtual T atImpl(const Address& address) const = 0;
  virtual void setImpl(const Address& address, T value,
                       std::function<T(T, T)> fn) = 0;
  virtual AddressIterator beginImpl() const = 0;
  virtual AddressIterator endImpl() const = 0;
};

}  // Tensor

#endif  // TENSOR_TENSOR_BASE_H

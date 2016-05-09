#ifndef SONNET_TENSOR_SHAPE_H
#define SONNET_TENSOR_SHAPE_H

#include <vector>
#include <initializer_list>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-conversion"
#include "glog/logging.h"
#pragma clang diagnostic pop

#include "util/serializable.h"

namespace Tensor {

// The tensor shape.
//
// A tensor has n dimensions.  There are two types of indices.
// Shape index refers associated with the index associate with dimension.
// And the address refers a accessing elements.

class Shape : public Util::Serializable {
 public:
  using Dims = std::vector<size_t>;
  using const_iterator = Dims::const_iterator;
  using const_reverse_iterator = Dims::const_reverse_iterator;

  Shape() {}
  explicit Shape(const Dims& dims);

  // Returns the number of dimensions.
  size_t nDimensions() const { return dims_.size(); }

  // Returns the size of dimension at dim_index.
  size_t operator[](size_t index) const { return dims_.at(index); }

  // Const iterators
  const_iterator begin() const { return dims_.cbegin(); }
  const_iterator end() const { return dims_.cend(); }

  const_iterator cbegin() const { return dims_.cbegin(); }
  const_iterator cend() const { return dims_.cend(); }

  // Const reverse iterators.
  const_reverse_iterator rbegin() const { return dims_.crbegin(); }
  const_reverse_iterator rend() const { return dims_.crend(); }

  const_reverse_iterator crbegin() const { return dims_.crbegin(); }
  const_reverse_iterator crend() const { return dims_.crend(); }

 private:
  using ArchiveIn = Util::ArchiveIn;
  using ArchiveOut = Util::ArchiveOut;
  void serializeInImpl(ArchiveIn& ar, size_t /*version*/) final { ar % dims_; }
  void serializeOutImpl(ArchiveOut& ar) const final { ar % dims_; }
  size_t serializeOutVersionImpl() const final { return 0ul; }

  Dims dims_;
};

// Are the two shapes equal?
inline bool operator==(const Shape& shape1, const Shape& shape2) {
  return shape1.nDimensions() == shape2.nDimensions() &&
         std::equal(shape1.cbegin(), shape1.cend(), shape2.cbegin());
}

// Are the two shapes not equal?
inline bool operator!=(const Shape& shape1, const Shape& shape2) {
  return !operator==(shape1, shape2);
}

// Calculate the number of elements for the shape.
size_t nElements(const Shape& shape);

// Indices to refer to Particular dimensions of shapes or address
using Indices = std::vector<int>;

// Addresses
using Address = std::vector<size_t>;


}  // Tensor
#endif

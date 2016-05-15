#include "neural_net/tensor/helpers.h"
#include "neural_net/tensor/shape.h"
#include <vector>

namespace NeuralNet {

namespace {
Shape resultShape(const Shape& shape1, const Indices& indices1,
                  const Shape& shape2, const Indices& indices2) {
  auto result_nDim =
      std::max(*std::max_element(indices1.cbegin(), indices1.cend()),
               *std::max_element(indices2.cbegin(), indices2.cend())) +
      1;

  std::vector<size_t> result_dim(static_cast<size_t>(result_nDim), 0);

  auto shape_pos = 0ul;
  for (const auto& index : indices1) {
    if (index >= 0) {
      auto cindex = static_cast<size_t>(index);
      result_dim[cindex] = shape1[shape_pos];
    }
    ++shape_pos;
  }

  shape_pos = 0ul;
  for (const auto& index : indices2) {
    if (index >= 0) {
      auto cindex = static_cast<size_t>(index);
      if (result_dim[cindex] != 0 && result_dim[cindex] != shape2[shape_pos]) {
        throw std::invalid_argument(
            "shared result indices refer to different shape sizes");
      }
      result_dim[cindex] = shape2[shape_pos];
    }
    ++shape_pos;
  }

  if (!std::all_of(result_dim.cbegin(), result_dim.cend(),
                   [](size_t dim) { return dim >= 1; })) {
    throw std::invalid_argument("union of indices are not consecutive");
  }

  return Shape(result_dim);
}

Shape commonShape(const Shape& shape1, const Indices& indices1,
                  const Shape& shape2, const Indices& indices2) {
  auto common_nDim =
      -std::min(*std::min_element(indices1.cbegin(), indices1.cend()),
                *std::min_element(indices2.cbegin(), indices2.cend()));

  std::vector<size_t> common_dim(static_cast<size_t>(common_nDim), 0);

  auto shape_pos = 0ul;
  for (const auto& index : indices1) {
    auto cindex = static_cast<size_t>(-index - 1);
    if (index < 0) {
      common_dim[cindex] = shape1[shape_pos];
    }
    ++shape_pos;
  }

  shape_pos = 0ul;
  for (const auto& index : indices2) {
    auto cindex = static_cast<size_t>(-index - 1);
    if (index < 0) {
      if (common_dim[cindex] == 0) {
        throw std::invalid_argument("non-repeated common index found");
      }
      if (common_dim[cindex] != shape2[shape_pos]) {
        throw std::invalid_argument(
            "(shared) common dim indices refer to different shape sizes");
      }
    }
    ++shape_pos;
  }

  if (!std::all_of(common_dim.cbegin(), common_dim.cend(),
                   [](size_t dim) { return dim >= 1; })) {
    throw std::invalid_argument("union of indices are not consecutive");
  }
  return Shape(common_dim);
}
}  // namespace

std::pair<Shape, Shape> multiplyShapes(const Shape& shape1,
                                       const Indices& indices1,
                                       const Shape& shape2,
                                       const Indices& indices2) {
  if (shape1.nDimensions() == 0 || shape1.nDimensions() != indices1.size())
    throw std::invalid_argument(
        "dimensions of shape1 and indices1 are inconsistent");
  if (shape2.nDimensions() == 0 || shape2.nDimensions() != indices2.size())
    throw std::invalid_argument(
        "dimensions of shape2 and indices2 are inconsistent");
  if (!indicesUnique(indices1))
    throw std::invalid_argument("indices1 are not unique");
  if (!indicesUnique(indices2))
    throw std::invalid_argument("indices2 are not unique");

  return {resultShape(shape1, indices1, shape2, indices2),
          commonShape(shape1, indices1, shape2, indices2)};
}

Shape combineShapes(const Shape& shape1, const Shape& shape2) {
  std::vector<size_t> dims(shape1.nDimensions() + shape2.nDimensions());
  auto iter = std::copy(shape1.cbegin(), shape1.cend(), dims.begin());
  std::copy(shape2.cbegin(), shape2.cend(), iter);
  return Shape(dims);
}

bool indicesUnique(Indices indices) {
  using namespace std;
  sort(indices.begin(), indices.end());
  return adjacent_find(indices.cbegin(), indices.cend()) == indices.cend();
}

}  // namespace Tensor

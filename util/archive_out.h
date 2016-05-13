#ifndef UTIL_ARCHIVE_OUT_H_
#define UTIL_ARCHIVE_OUT_H_

#include <array>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Util {

/* Archive class for serialisation. */
class ArchiveOut {
 public:
  explicit ArchiveOut(std::ostream* stream) : stream_(stream) {}
  ~ArchiveOut() {}

  // Serialize primitives and generic classes.
  template <typename T>
  ArchiveOut& operator%(const T& value);

  // Serialize std::pairs
  template <typename TFirst, typename TSecond>
  ArchiveOut& operator%(const std::pair<TFirst, TSecond>& value);

  // Serialize std::array.
  template <typename TValue, size_t n>
  ArchiveOut& operator%(const std::array<TValue, n>& value);

  // Serialize std::vector.
  template <typename TValue>
  ArchiveOut& operator%(const std::vector<TValue>& value);

  // Serialize std::unordered_set.
  template <typename TValue, typename THash, typename TEqual>
  ArchiveOut& operator%(
      const std::unordered_set<TValue, THash, TEqual>& container);

  // Serialize std::unordered_map.
  template <typename TKey, typename TValue, typename THash, typename TEqual>
  ArchiveOut& operator%(
      const std::unordered_map<TKey, TValue, THash, TEqual>& container);

  // Serialize std::map.
  template <typename TKey, typename TValue, typename TCompare>
  ArchiveOut& operator%(const std::map<TKey, TValue, TCompare>& container);

  // Note: other standard containers to be added as necessary.

 private:
  // Convenient private function for reading primitives.
  template <typename T>
  void writePrimitive(const T& value);

  std::ostream* stream_;
};

template <typename TContainer>
ArchiveOut& serializeOutContainer(ArchiveOut& ar, const TContainer& container) {
  using value_type = typename TContainer::value_type;

  ar % container.size();
  std::for_each(container.cbegin(), container.cend(),
                [&ar](const value_type& value) { ar % value; });

  return ar;
}

template <typename T>
ArchiveOut& ArchiveOut::operator%(const T& value) {
  value.serializeOut(*this);
  return *this;
}

template <typename T>
void ArchiveOut::writePrimitive(const T& value) {
  stream_->write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename TFirst, typename TSecond>
ArchiveOut& ArchiveOut::operator%(const std::pair<TFirst, TSecond>& value) {
  return (*this) % value.first % value.second;
}

template <typename TValue, size_t n>
ArchiveOut& ArchiveOut::operator%(const std::array<TValue, n>& container) {
  return serializeOutContainer(*this, container);
}

template <typename TValue>
ArchiveOut& ArchiveOut::operator%(const std::vector<TValue>& container) {
  return serializeOutContainer(*this, container);
}

template <typename TValue, typename THash, typename TEqual>
ArchiveOut& ArchiveOut::operator%(
    const std::unordered_set<TValue, THash, TEqual>& container) {
  return serializeOutContainer(*this, container);
}

template <typename TKey, typename TValue, typename THash, typename TEqual>
ArchiveOut& ArchiveOut::operator%(
    const std::unordered_map<TKey, TValue, THash, TEqual>& container) {
  return serializeOutContainer(*this, container);
}

template <typename TKey, typename TValue, typename TCompare>
ArchiveOut& ArchiveOut::operator%(
    const std::map<TKey, TValue, TCompare>& container) {
  return serializeOutContainer(*this, container);
}

template <>
ArchiveOut& ArchiveOut::operator%(const bool& value);

template <>
ArchiveOut& ArchiveOut::operator%(const char& value);

template <>
ArchiveOut& ArchiveOut::operator%(const signed char& value);

template <>
ArchiveOut& ArchiveOut::operator%(const unsigned char& value);

template <>
ArchiveOut& ArchiveOut::operator%(const short int& value);

template <>
ArchiveOut& ArchiveOut::operator%(const unsigned short int& value);

template <>
ArchiveOut& ArchiveOut::operator%(const int& value);

template <>
ArchiveOut& ArchiveOut::operator%(const unsigned int& value);

template <>
ArchiveOut& ArchiveOut::operator%(const long int& value);

template <>
ArchiveOut& ArchiveOut::operator%(const unsigned long int& value);

template <>
ArchiveOut& ArchiveOut::operator%(const float& value);

template <>
ArchiveOut& ArchiveOut::operator%(const double& value);

template <>
ArchiveOut& ArchiveOut::operator%(const long double& value);

template <>
ArchiveOut& ArchiveOut::operator%(const std::string& value);

}  // namespace Util

#endif  // UTIL_ARCHIVE_OUT_H_

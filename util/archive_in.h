#ifndef UTIL_ARCHIVE_IN_H_
#define UTIL_ARCHIVE_IN_H_

#include <iostream>
#include <array>
#include <string>
#include <vector>
#include <unordered_set>

namespace Util {

/* Archive class for serialisation. */
class ArchiveIn {
 public:
  explicit ArchiveIn(std::istream* stream) : stream_(stream) {}
  ~ArchiveIn() {}

  // Serialize primitives and generic classes.
  template <typename T>
  ArchiveIn& operator%(T& value);

  // Serialize std::array.
  template <typename TValue, size_t size>
  ArchiveIn& operator%(std::array<TValue, size>& container);

  // Serialize std::vector.
  template <typename TValue>
  ArchiveIn& operator%(std::vector<TValue>& container);

  // Serialize std::unordered_set.
  template <typename TValue, typename THash, typename TEqual>
  ArchiveIn& operator%(std::unordered_set<TValue, THash, TEqual>& container);

  // Note: other standard containers to be added as necessary.

 private:
  // Convenient private function for reading primitives.
  template <typename T>
  void readPrimitive(T& value);

  std::istream* stream_;
};

// Convenience function to serialize value containers.
template <typename TContainer>
ArchiveIn& serializeInValueContainer(
    ArchiveIn& ar, std::function<void(typename TContainer::value_type)> insert,
    std::function<void()> clear,
    std::function<void(size_t)> reserve = [](size_t) {}) {
  auto size = 0ul;
  ar % size;

  clear();
  reserve(size);

  for (auto index = 0ul; index < size; ++index) {
    typename TContainer::value_type v;
    ar % v;
    insert(v);
  }

  return ar;
}

template <typename T>
ArchiveIn& ArchiveIn::operator%(T& value) {
  value.serializeIn(*this);
  return *this;
}

template <typename T>
void ArchiveIn::readPrimitive(T& value) {
  static char temp[sizeof(T)];
  stream_->read(temp, sizeof(T));
  value = *reinterpret_cast<T*>(temp);
}

template <typename TValue, size_t size>
ArchiveIn& ArchiveIn::operator%(std::array<TValue, size>& container) {
  auto index = 0ul;
  return serializeInValueContainer<std::array<TValue, size>>(
      *this, [&container, &index](TValue value) { container[index++] = value; },
      []() {});
}

template <typename TValue>
ArchiveIn& ArchiveIn::operator%(std::vector<TValue>& container) {
  return serializeInValueContainer<std::vector<TValue>>(
      *this, [&container](TValue value) { container.emplace_back(value); },
      [&container]() { container.clear(); },
      [&container](size_t size) { container.reserve(size); });
}

template <typename TValue, typename THash, typename TEqual>
ArchiveIn& ArchiveIn::operator%(
    std::unordered_set<TValue, THash, TEqual>& container) {
  return serializeInValueContainer<std::unordered_set<TValue, THash, TEqual>>(
      *this, [&container](TValue value) { container.emplace(value); },
      [&container](void) { container.clear(); });
}

template <>
ArchiveIn& ArchiveIn::operator%(bool& value);

template <>
ArchiveIn& ArchiveIn::operator%(char& value);

template <>
ArchiveIn& ArchiveIn::operator%(signed char& value);

template <>
ArchiveIn& ArchiveIn::operator%(unsigned char& value);

template <>
ArchiveIn& ArchiveIn::operator%(short int& value);

template <>
ArchiveIn& ArchiveIn::operator%(unsigned short int& value);

template <>
ArchiveIn& ArchiveIn::operator%(int& value);

template <>
ArchiveIn& ArchiveIn::operator%(unsigned int& value);

template <>
ArchiveIn& ArchiveIn::operator%(long int& value);

template <>
ArchiveIn& ArchiveIn::operator%(unsigned long int& value);

template <>
ArchiveIn& ArchiveIn::operator%(float& value);

template <>
ArchiveIn& ArchiveIn::operator%(double& value);

template <>
ArchiveIn& ArchiveIn::operator%(long double& value);

template <>
ArchiveIn& ArchiveIn::operator%(std::string& value);

}  // namespace Util

#endif  // UTIL_ARCHIVE_IN_H_

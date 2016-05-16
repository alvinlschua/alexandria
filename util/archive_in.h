#ifndef UTIL_ARCHIVE_IN_H_
#define UTIL_ARCHIVE_IN_H_

#include <array>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Alexandria {

/* Archive class for serialisation. */
class ArchiveIn {
 public:
  explicit ArchiveIn(std::istream* stream) : stream_(stream) {}
  ~ArchiveIn() {}

  // Serialize primitives and generic classes.
  template <typename T>
  ArchiveIn& operator%(T& value);

  // Serialize std::pair.
  template <typename TFirst, typename TSecond>
  ArchiveIn& operator%(std::pair<TFirst, TSecond>& container);

  // Serialize std::array.
  template <typename TValue, size_t size>
  ArchiveIn& operator%(std::array<TValue, size>& container);

  // Serialize std::vector.
  template <typename TValue>
  ArchiveIn& operator%(std::vector<TValue>& container);

  // Serialize std::unordered_set.
  template <typename TValue, typename THash, typename TEqual>
  ArchiveIn& operator%(std::unordered_set<TValue, THash, TEqual>& container);

  // Serialize std::unordered_map.
  template <typename TKey, typename TValue, typename THash, typename TEqual>
  ArchiveIn& operator%(
      std::unordered_map<TKey, TValue, THash, TEqual>& container);

  // Serialize std::unordered_map.
  template <typename TKey, typename TValue, typename TCompare>
  ArchiveIn& operator%(std::map<TKey, TValue, TCompare>& container);

  // Note: other standard containers to be added as necessary.

 private:
  // Convenient private function for reading primitives.
  template <typename T>
  void readPrimitive(T& value);

  std::istream* stream_;
};

// Convenience function to serialize value containers.
template <typename TContainer, typename TValue>
ArchiveIn& serializeInContainer(ArchiveIn& ar,
                                std::function<void(TValue)> insert,
                                std::function<void()> clear,
                                std::function<void(size_t)> reserve =
                                    [](size_t) {}) {
  auto size = 0ul;
  ar % size;

  clear();
  reserve(size);

  for (auto index = 0ul; index < size; ++index) {
    TValue value;
    ar % value;
    insert(value);
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

template <typename TFirst, typename TSecond>
ArchiveIn& ArchiveIn::operator%(std::pair<TFirst, TSecond>& value) {
  return (*this) % value.first % value.second;
}

template <typename TValue, size_t size>
ArchiveIn& ArchiveIn::operator%(std::array<TValue, size>& container) {
  using ContainerType = std::array<TValue, size>;
  using ValueType = typename ContainerType::value_type;

  auto index = 0ul;
  return serializeInContainer<ContainerType, ValueType>(
      *this,
      [&container, &index](const TValue& value) { container[index++] = value; },
      []() {});
}

template <typename TValue>
ArchiveIn& ArchiveIn::operator%(std::vector<TValue>& container) {
  using ContainerType = std::vector<TValue>;
  using ValueType = typename ContainerType::value_type;

  return serializeInContainer<ContainerType, ValueType>(
      *this,
      [&container](const TValue& value) { container.emplace_back(value); },
      [&container]() { container.clear(); },
      [&container](size_t size) { container.reserve(size); });
}

template <typename TValue, typename THash, typename TEqual>
ArchiveIn& ArchiveIn::operator%(
    std::unordered_set<TValue, THash, TEqual>& container) {
  using ContainerType = std::unordered_set<TValue, THash, TEqual>;
  using ValueType = typename ContainerType::value_type;

  return serializeInContainer<ContainerType, ValueType>(
      *this, [&container](const ValueType& value) { container.emplace(value); },
      [&container](void) { container.clear(); },
      [&container](size_t size) { container.reserve(size); });
}

template <typename TKey, typename TValue, typename THash, typename TEqual>
ArchiveIn& ArchiveIn::operator%(
    std::unordered_map<TKey, TValue, THash, TEqual>& container) {
  using ContainerType = std::unordered_map<TKey, TValue, THash, TEqual>;
  using ValueType = std::pair<TKey, TValue>;

  return serializeInContainer<ContainerType, ValueType>(
      *this, [&container](const ValueType& value) { container.emplace(value); },
      [&container](void) { container.clear(); },
      [&container](size_t size) { container.reserve(size); });
}

template <typename TKey, typename TValue, typename TCompare>
ArchiveIn& ArchiveIn::operator%(
    std::map<TKey, TValue, TCompare>& container) {
  using ContainerType = std::map<TKey, TValue, TCompare>;
  using ValueType = std::pair<TKey, TValue>;

  return serializeInContainer<ContainerType, ValueType>(
      *this, [&container](const ValueType& value) { container.emplace(value); },
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

}  // namespace Alexandria

#endif  // UTIL_ARCHIVE_IN_H_

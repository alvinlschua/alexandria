#include "util/archive_in.h"

namespace Alexandria {

template <>
ArchiveIn& ArchiveIn::operator%(bool& value) {
  readPrimitive(value);
  return *this;
}

template <>
ArchiveIn& ArchiveIn::operator%(char& value) {
  readPrimitive(value);
  return *this;
}

template <>
ArchiveIn& ArchiveIn::operator%(signed char& value) {
  readPrimitive(value);
  return *this;
}

template <>
ArchiveIn& ArchiveIn::operator%(unsigned char& value) {
  readPrimitive(value);
  return *this;
}

template <>
ArchiveIn& ArchiveIn::operator%(short int& value) {
  readPrimitive(value);
  return *this;
}

template <>
ArchiveIn& ArchiveIn::operator%(unsigned short int& value) {
  readPrimitive(value);
  return *this;
}

template <>
ArchiveIn& ArchiveIn::operator%(int& value) {
  readPrimitive(value);
  return *this;
}

template <>
ArchiveIn& ArchiveIn::operator%(unsigned int& value) {
  readPrimitive(value);
  return *this;
}

template <>
ArchiveIn& ArchiveIn::operator%(long int& value) {
  readPrimitive(value);
  return *this;
}

template <>
ArchiveIn& ArchiveIn::operator%(unsigned long int& value) {
  readPrimitive(value);
  return *this;
}

template <>
ArchiveIn& ArchiveIn::operator%(float& value) {
  readPrimitive(value);
  return *this;
}

template <>
ArchiveIn& ArchiveIn::operator%(double& value) {
  readPrimitive(value);
  return *this;
}

template <>
ArchiveIn& ArchiveIn::operator%(long double& value) {
  readPrimitive(value);
  return *this;
}

template <>
ArchiveIn& ArchiveIn::operator%(std::string& value) {
  size_t size = 0ul;
  (*this) % size;

  value.resize(size);
  stream_->read(&value[0], static_cast<long>(size));
  return *this;
}

}  // namespace Alexandria

#include "util/archive_out.h"

namespace Util {

template <>
ArchiveOut& ArchiveOut::operator%(const bool& value) {
  writePrimitive(value);
  return *this;
}

template <>
ArchiveOut& ArchiveOut::operator%(const char& value) {
  writePrimitive(value);
  return *this;
}

template <>
ArchiveOut& ArchiveOut::operator%(const signed char& value) {
  writePrimitive(value);
  return *this;
}

template <>
ArchiveOut& ArchiveOut::operator%(const unsigned char& value) {
  writePrimitive(value);
  return *this;
}

template <>
ArchiveOut& ArchiveOut::operator%(const short int& value) {
  writePrimitive(value);
  return *this;
}

template <>
ArchiveOut& ArchiveOut::operator%(const unsigned short int& value) {
  writePrimitive(value);
  return *this;
}

template <>
ArchiveOut& ArchiveOut::operator%(const int& value) {
  writePrimitive(value);
  return *this;
}

template <>
ArchiveOut& ArchiveOut::operator%(const unsigned int& value) {
  writePrimitive(value);
  return *this;
}

template <>
ArchiveOut& ArchiveOut::operator%(const long int& value) {
  writePrimitive(value);
  return *this;
}

template <>
ArchiveOut& ArchiveOut::operator%(const unsigned long int& value) {
  writePrimitive(value);
  return *this;
}

template <>
ArchiveOut& ArchiveOut::operator%(const float& value) {
  writePrimitive(value);
  return *this;
}

template <>
ArchiveOut& ArchiveOut::operator%(const double& value) {
  writePrimitive(value);
  return *this;
}

template <>
ArchiveOut& ArchiveOut::operator%(const long double& value) {
  writePrimitive(value);
  return *this;
}

template <>
ArchiveOut& ArchiveOut::operator%(const std::string& value) {
  (*this) % value.size();
  stream_->write(reinterpret_cast<const char*>(&value[0]),
                 static_cast<long>(value.size()));
  return *this;
}

}  // namespace Util

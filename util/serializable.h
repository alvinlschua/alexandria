#ifndef UTIL_SERIALIZABLE_H
#define UTIL_SERIALIZABLE_H

#include "util/archive_in.h"
#include "util/archive_out.h"

namespace Util {

// Serializable interface.
//
// Requires concrete instances of the base class to be serializable.
// A default constructor should be provided by the class.
class Serializable {
 public:
  virtual ~Serializable() {}
  Serializable() = default;
  Serializable(const Serializable&) = default;
  Serializable& operator=(const Serializable&) = default;

  // Serialize in method called by archive.
  void serializeIn(ArchiveIn& ar) {
    auto version = 0ul;
    ar % version;
    serializeInImpl(ar, version);
  }

  // Serialize out method called by archive.
  void serializeOut(ArchiveOut& ar) const {
    ar % serializeOutVersion();
    serializeOutImpl(ar);
  }

  // Returns serialize out version.
  size_t serializeOutVersion() const { return serializeOutVersionImpl(); }

 private:
  // version denotes the version number being read in.
  virtual void serializeInImpl(ArchiveIn& ar, size_t version) = 0;
  virtual void serializeOutImpl(ArchiveOut& ar) const = 0;
  virtual size_t serializeOutVersionImpl() const = 0;
};

}  // namespace Util

#endif

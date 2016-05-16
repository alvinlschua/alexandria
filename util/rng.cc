#include "util/rng.h"

#include <string>

namespace Alexandria {

void Rng::serializeInImpl(ArchiveIn& ar, size_t /*version*/) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto state = std::string();
  ar % state;
  std::istringstream sin(state);
  sin >> generator_;
}

void Rng::serializeOutImpl(ArchiveOut& ar) const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::ostringstream sout;
  sout << generator_;
  ar % sout.str();
}

}  // namespace Alexandria

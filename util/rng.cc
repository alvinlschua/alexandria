#include "util/rng.h"

namespace Util {

void Rng::serializeInImpl(ArchiveIn& ar, size_t /*version*/) {
  using namespace std;
  lock_guard<mutex> lock(mutex_);
  auto state = string();
  ar % state;
  istringstream sin(state);
  sin >> generator_;
}

void Rng::serializeOutImpl(ArchiveOut& ar) const {
  using namespace std;
  lock_guard<mutex> lock(mutex_);
  ostringstream sout;
  sout << generator_;
  ar % sout.str();
}

}  // Util

#ifndef UTIL_SINGLETON_H_
#define UTIL_SINGLETON_H_

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-conversion"
#include "glog/logging.h"
#pragma clang diagnostic pop

namespace Alexandria {

// Generic Singleton class.
//
// Type T must be the class itself.
// E.g. class objectStore : public Singleton<objectStore> { ... };
template <typename T>
class Singleton {
 public:
  virtual ~Singleton() {}

  // Return the instance of the class.
  static T& instance();

  // Reset the instance.
  static void reset() { m_instance.reset(); }

 protected:
  // Prevents the default constructor from being called after the singleton has
  // been first instantiated
  Singleton() { CHECK(!m_instance) << "instance already there"; }

  // Populate the instance if necessary. Acts as a constructor for the
  // singleton.
  virtual void populate() {}

 private:
  Singleton(const Singleton&);
  Singleton operator=(const Singleton&);

  static std::unique_ptr<T> m_instance;
};

template <typename T>
T& Singleton<T>::instance() {
  if (m_instance.get() == 0) {
    m_instance = std::make_unique<T>();
    m_instance->populate();
  }

  return *m_instance;
}

template <typename T>
std::unique_ptr<T> Singleton<T>::m_instance = std::unique_ptr<T>();

}  // namespace Alexandria

#endif  // UTIL_SINGLETON_H_

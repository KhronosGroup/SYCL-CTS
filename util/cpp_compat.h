/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides advanced features not available in C++17
//
*******************************************************************************/

#include <map>
#include <unordered_set>

#if __cplusplus >= 202002L
#define SYCL_CTS_COMPAT_CPP20 [[deprecated]]
#else
#define SYCL_CTS_COMPAT_CPP20
#endif

namespace sycl_cts::util {

namespace details {
/** @brief Provides implementation for std::erase_if
 */
template <class Container, class Pred>
inline typename Container::size_type erase_if(Container& container, Pred pred) {
  auto old_size = container.size();
  // In-place conditional erasure for container
  for (auto it = container.begin(); it != container.end();) {
    if (pred(it)) {
      it = container.erase(it);
    } else {
      ++it;
    }
  }
  return old_size - container.size();
}
}  // namespace details

/** @brief Provides std::erase_if from std::map available from c++20
 */
template <class Key, class T, class Compare, class Alloc, class Pred>
SYCL_CTS_COMPAT_CPP20 static typename std::map<Key, T, Compare, Alloc>::size_type
erase_if(std::map<Key, T, Compare, Alloc>& map, Pred pred) {
  return details::erase_if(map, pred);
}

/** @brief Provides std::erase_if form std::unordered_set available from c++20
 */
template <class Key, class Hash, class KeyEqual, class Alloc, class Pred>
SYCL_CTS_COMPAT_CPP20 static
    typename std::unordered_set<Key, Hash, KeyEqual, Alloc>::size_type
    erase_if(std::unordered_set<Key, Hash, KeyEqual, Alloc>& set, Pred pred) {
  return details::erase_if(set, pred);
}

}  // namespace sycl_cts::util

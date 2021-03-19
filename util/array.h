/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provide fixed-width array implementation for usage in kernel functions
//  because std::array methods are not required to be noexcept by C++ spec,
//  while exceptions are not allowed in a device function
//
*******************************************************************************/

#ifndef __SYCLCTS_UTIL_ARRAY_H
#define __SYCLCTS_UTIL_ARRAY_H

namespace sycl_cts {
namespace util {

/** @brief Fixed width array to use in kernel functions
 */
template <class T, std::size_t N>
struct array {
  // Should be aggregate type, so no explicit constructors
  // and private attributes

  using value_type = T;
  using reference = value_type&;
  using const_reference = const value_type&;
  using size_type = std::size_t;

  /** @brief Iterator type; is safe to use on host or device side only
   */
  using iterator = value_type*;
  using const_iterator = const value_type*;

  /** @brief Internal data storage;
   *         public for aggregate initialization availability
   */
  value_type values[N];

  reference operator[](size_type pos) {
    return values[pos];
  }
  const_reference operator[](size_type pos) const {
    return values[pos];
  }

  constexpr size_type size() const noexcept {
    return N;
  }

  reference front() {
    return values[0];
  }
  const_reference front() const {
    return values[0];
  }
  reference back() {
    return values[N-1];
  }
  const_reference back() const {
    return values[N-1];
  }

  /** @brief Support for C++ ranged-for syntax
   */
  iterator begin() noexcept {
    return &front();
  }
  const_iterator begin() const noexcept {
    return &front();
  }
  iterator end() noexcept {
    return &back();
  }
  const_iterator end() const noexcept {
    return &back();
  }
};

}  // namespace util
}  // namespace sycl_cts

#endif  // __SYCLCTS_TESTS_COMMON_ARRAY_H

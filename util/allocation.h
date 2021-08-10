/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provide union that let allocate memory without default constructor calling
//
*******************************************************************************/

#ifndef __SYCL_UTIL_ALLOCATION_H
#define __SYCL_UTIL_ALLOCATION_H

namespace sycl_cts {
namespace util {

template <typename T>
union remove_initialization {
  using value_type = T;
  T value;
  remove_initialization() {}

  template <typename T>
  void operator=(const T &val) {
    this->value = val;
  }

  friend bool operator==(const remove_initialization &lhs,
                         const remove_initialization &rhs) {
    return lhs.value == rhs.value;
  }

  operator value_type &() { return value; }
  operator const value_type &() const { return value; }

  value_type *data() { return &value; }
  const value_type *data() const { return &value; }
};

}  // namespace util
}  // namespace sycl_cts

#endif  // __SYCLCTS_UTIL_ACCURACY_H

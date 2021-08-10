/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Common functions for multi_ptr tests
//
*******************************************************************************/

#ifndef SYCL_CTS_TEST_MULTI_PTR_MULTI_PTR_COMMON_H
#define SYCL_CTS_TEST_MULTI_PTR_MULTI_PTR_COMMON_H

#include "../common/common.h"

namespace multi_ptr_common {

template <typename... argsT>
void silence_warnings(argsT...) {
  // Just to avoid compiler warnings on unused variables
}

/** @brief Wrapper with type pairs for multi_ptr with 'void' type verification
 */
template <template <typename, typename> class action, typename T>
struct check_void_pointer {
  using data_t = typename std::remove_const<T>::type;

  template <typename... argsT>
  void operator()(argsT&&... args) {
    action<data_t, void>{}(std::forward<argsT>(args)..., "void");
    action<const data_t, const void>{}(std::forward<argsT>(args)..., "void");
  }
};

/** @brief Wrapper with type pairs for generic multi_ptr verification
 */
template <template <typename, typename> class action, typename T>
struct check_pointer {
  using data_t = typename std::remove_const<T>::type;

  template <typename... argsT>
  void operator()(argsT&&... args) {
    action<data_t, data_t>{}(std::forward<argsT>(args)...);
    action<const data_t, const data_t>{}(std::forward<argsT>(args)...);
  }
};

}  // namespace multi_ptr_common

#endif  // SYCL_CTS_TEST_MULTI_PTR_MULTI_PTR_COMMON_H

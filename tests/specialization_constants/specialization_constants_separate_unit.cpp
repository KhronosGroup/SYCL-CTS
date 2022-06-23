/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides separate unit for checks for specialization constants
//  with SYCL_EXTERNAL function
//
*******************************************************************************/

#include "../common/common.h"
#include "specialization_constants_common.h"

using namespace get_spec_const;

template <typename T, int case_num>
inline constexpr sycl::specialization_id<T> spec_const_external(
    user_def_types::get_init_value_helper<T>(default_val));

template <typename T, test_cases_external TestCase>
bool check_kernel_handler_by_reference_external(sycl::kernel_handler &h,
                                                T expected) {
  constexpr int case_num = static_cast<int>(TestCase);
  return check_equal_values(
      expected,
      h.get_specialization_constant<spec_const_external<T, case_num>>());
}

template <typename T, test_cases_external TestCase>
bool check_kernel_handler_by_value_external(sycl::kernel_handler h,
                                            T expected) {
  constexpr int case_num = static_cast<int>(TestCase);
  return check_equal_values(
      expected,
      h.get_specialization_constant<spec_const_external<T, case_num>>());
}

#define FUNC_DEFINE(TYPE)                                                      \
                                                                               \
  SYCL_EXTERNAL bool check_kernel_handler_by_reference_external_handler(       \
      sycl::kernel_handler &h, TYPE expected) {                                \
    return check_kernel_handler_by_reference_external<                         \
        TYPE, test_cases_external::by_reference_via_handler>(h, expected);     \
  }                                                                            \
                                                                               \
  SYCL_EXTERNAL bool check_kernel_handler_by_value_external_handler(           \
      sycl::kernel_handler h, TYPE expected) {                                 \
    return check_kernel_handler_by_value_external<                             \
        TYPE, test_cases_external::by_value_via_handler>(h, expected);         \
  }                                                                            \
                                                                               \
  SYCL_EXTERNAL bool check_kernel_handler_by_reference_external_bundle(        \
      sycl::kernel_handler &h, TYPE expected) {                                \
    return check_kernel_handler_by_reference_external<                         \
        TYPE, test_cases_external::by_reference_via_bundle>(h, expected);      \
  }                                                                            \
                                                                               \
  SYCL_EXTERNAL bool check_kernel_handler_by_value_external_bundle(            \
      sycl::kernel_handler h, TYPE expected) {                                 \
    return check_kernel_handler_by_value_external<                             \
        TYPE, test_cases_external::by_value_via_bundle>(h, expected);          \
  }

#if !SYCL_CTS_ENABLE_FULL_CONFORMANCE
CORE_TYPES(FUNC_DEFINE)
#else
CORE_TYPES_PARAM(SYCL_VECTORS_MARRAYS, FUNC_DEFINE)
#endif

FUNC_DEFINE(user_def_types::no_cnstr)
FUNC_DEFINE(user_def_types::def_cnstr)
FUNC_DEFINE(user_def_types::no_def_cnstr)

#if SYCL_CTS_ENABLE_DOUBLE_TESTS
#if !SYCL_CTS_ENABLE_FULL_CONFORMANCE
FUNC_DEFINE(double)
#else
SYCL_VECTORS_MARRAYS(double, FUNC_DEFINE)
#endif
#endif  // SYCL_CTS_ENABLE_DOUBLE_TESTS

#if SYCL_CTS_ENABLE_HALF_TESTS
#if !SYCL_CTS_ENABLE_FULL_CONFORMANCE
FUNC_DEFINE(sycl::half)
#else
SYCL_VECTORS_MARRAYS(sycl::half, FUNC_DEFINE)
#endif
#endif  // SYCL_CTS_ENABLE_HALF_TESTS

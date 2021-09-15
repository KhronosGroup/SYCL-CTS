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
    value_helper<T>(default_val));

template <typename T, int case_num>
bool check_kernel_handler_by_reference_external(sycl::kernel_handler &h,
                                                T expected) {
  return check_equal_values(
      expected,
      h.get_specialization_constant<spec_const_external<T, case_num>>());
}

template <typename T, int case_num>
bool check_kernel_handler_by_value_external(sycl::kernel_handler h,
                                            T expected) {
  return check_equal_values(
      expected,
      h.get_specialization_constant<spec_const_external<T, case_num>>());
}

#define FUNC_DEFINE(TYPE)                                                      \
                                                                               \
  SYCL_EXTERNAL bool check_kernel_handler_by_reference_external_handler(       \
      sycl::kernel_handler &h, TYPE expected) {                                \
    return check_kernel_handler_by_reference_external<                         \
        TYPE, by_reference_via_handler>(h, expected);                          \
  }                                                                            \
                                                                               \
  SYCL_EXTERNAL bool check_kernel_handler_by_value_external_handler(           \
      sycl::kernel_handler h, TYPE expected) {                                 \
    return check_kernel_handler_by_value_external<TYPE, by_value_via_handler>( \
        h, expected);                                                          \
  }                                                                            \
                                                                               \
  SYCL_EXTERNAL bool check_kernel_handler_by_reference_external_bundle(        \
      sycl::kernel_handler &h, TYPE expected) {                                \
    return check_kernel_handler_by_reference_external<                         \
        TYPE, by_reference_via_bundle>(h, expected);                           \
  }                                                                            \
                                                                               \
  SYCL_EXTERNAL bool check_kernel_handler_by_value_external_bundle(            \
      sycl::kernel_handler h, TYPE expected) {                                 \
    return check_kernel_handler_by_value_external<TYPE, by_value_via_bundle>(  \
        h, expected);                                                          \
  }

#ifndef SYCL_CTS_FULL_CONFORMANCE
CORE_TYPES(FUNC_DEFINE)
#else
CORE_TYPES_PARAM(SYCL_VECTORS_MARRAYS, FUNC_DEFINE)
#endif

FUNC_DEFINE(testing_types::no_cnstr)
FUNC_DEFINE(testing_types::def_cnstr)
FUNC_DEFINE(testing_types::no_def_cnstr)

#ifdef SYCL_CTS_TEST_DOUBLE
#ifndef SYCL_CTS_FULL_CONFORMANCE
FUNC_DEFINE(double)
#else
SYCL_VECTORS_MARRAYS(double, FUNC_DEFINE)
#endif
#endif  // SYCL_CTS_TEST_DOUBLE

#ifdef SYCL_CTS_TEST_HALF
#ifndef SYCL_CTS_FULL_CONFORMANCE
FUNC_DEFINE(sycl::half)
#else
SYCL_VECTORS_MARRAYS(sycl::half, FUNC_DEFINE)
#endif
#endif  // SYCL_CTS_TEST_HALF

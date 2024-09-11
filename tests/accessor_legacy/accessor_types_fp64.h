/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provide common code for accessor verification with fp64 types
//
*******************************************************************************/

#ifndef SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_TYPES_FP64_H
#define SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_TYPES_FP64_H

#include "../common/common.h"
#include "../common/type_coverage.h"
#include "./../../util/extensions.h"

#ifndef TEST_NAME
#error Invalid test namespace
#endif

namespace TEST_NAMESPACE {

template <typename T>
struct kernel_name {};

// Nested struct type usage in kernel name will be deprecated in SYCL 2020
// These tests should be able to verify accessor data types without dependency
// on kernel name restrictions
struct nested_struct_kernel {};

/**
 *  @brief Run specific accessors' tests for fp16 type set for generic code path
 */
template <template <typename, typename, typename> class action>
class check_all_types_fp64 {

  using extension_tag_t = sycl_cts::util::extensions::tag::fp64;

  template <typename T>
  using check_type = action<T, extension_tag_t, kernel_name<T>>;

public:
  static void run(sycl::queue& queue, sycl_cts::util::logger &log) {

    // Skip tests in case extension is not available
    using availability =
        sycl_cts::util::extensions::availability<extension_tag_t>;
    if (!availability::check(queue, log))
      return;

#if !SYCL_CTS_ENABLE_FULL_CONFORMANCE
    // Specific set of types to cover during ordinary compilation

    /** check specific accessor api for double
     */
    check_type<double>()(log, queue, "double");
    /** check specific accessor api for vec
     */
    check_type<sycl::double3>()(log, queue, "double");
#else
    // Extended type coverage
    for_type_and_vectors<check_type, double>(
        log, queue, "double");
    for_type_and_vectors<check_type, sycl::opencl::cl_double>(
        log, queue, "sycl::opencl::cl_double");

#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE

    queue.wait_and_throw();
  }
};

}  // namespace TEST_NAMESPACE

#endif // SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_TYPES_FP64_H

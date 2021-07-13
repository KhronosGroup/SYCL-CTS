/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provide common code for accessor verification with fp16 types
//
*******************************************************************************/

#ifndef SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_TYPES_FP16_H
#define SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_TYPES_FP16_H

#include "../common/common.h"
#include "../common/type_coverage.h"
#include "./../../util/extensions.h"

#ifndef TEST_NAME
#error Invalid test namespace
#endif

namespace TEST_NAMESPACE {

template <typename T>
struct kernel_name {};

/**
 *  @brief Run specific accessors' tests for fp16 type set for generic code path
 */
template <template <typename, typename, typename> class action>
class check_all_types_fp16 {

  using extension_tag_t = sycl_cts::util::extensions::tag::fp16;

  template <typename T>
  using check_type = action<T, extension_tag_t, kernel_name<T>>;

public:
  static void run(sycl::queue& queue, sycl_cts::util::logger &log) {

    // Skip tests in case extension is not available
    using availability =
        sycl_cts::util::extensions::availability<extension_tag_t>;
    if (!availability::check(queue, log))
      return;

#ifndef SYCL_CTS_FULL_CONFORMANCE
    // Specific set of types to cover during ordinary compilation

    /** check specific accessor api for half
     */
    check_type<sycl::half>()(log, queue, "sycl::half");
    /** check specific accessor api for vec
     */
    check_type<sycl::half3>()(log, queue, "sycl::half");
#else
    // Extended type coverage
    for_type_and_vectors<check_type, sycl::half>(
        log, queue, "sycl::half");
    for_type_and_vectors<check_type, sycl::cl_half>(
        log, queue, "sycl::cl_half");

#endif // SYCL_CTS_FULL_CONFORMANCE

    queue.wait_and_throw();
  }
};

}  // namespace TEST_NAMESPACE

#endif // SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_TYPES_FP16_H

/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provide common code for accessor API verification with fp64 types
//
*******************************************************************************/

#ifndef SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_API_TYPES_FP64_H
#define SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_API_TYPES_FP64_H

#include "../common/common.h"
#include "../common/type_coverage.h"
#include "./../../util/extensions.h"

#ifndef TEST_NAME
#error Invalid test namespace
#endif

namespace TEST_NAMESPACE {

/**
 *  @brief Run specific accessors' tests for fp16 type set for generic code path
 */
template <template <typename, typename> class action>
class check_all_types_fp64 {

  using extension_tag_t = sycl_cts::util::extensions::tag::fp64;

  template <typename T>
  using check_type = action<T, extension_tag_t>;

public:
  static void run(cl::sycl::queue& queue, sycl_cts::util::logger &log) {

    // Skip tests in case extension is not available
    using availability =
        sycl_cts::util::extensions::availability<extension_tag_t>;
    if (!availability::check(queue, log))
      return;

#ifndef SYCL_CTS_FULL_CONFORMANCE
    // Specific set of types to cover during ordinary compilation

    /** check specific accessor api for double
     */
    check_type<double>()(log, queue, "double");
    /** check specific accessor api for vec
     */
    check_type<cl::sycl::double3>()(log, queue, "double");
#else
    // Extended type coverage
    for_type_and_vectors<check_type, double>(
        log, queue, "double");
    for_type_and_vectors<check_type, cl::sycl::cl_double>(
        log, queue, "cl::sycl::cl_double");

#endif // SYCL_CTS_FULL_CONFORMANCE

    queue.wait_and_throw();
  }
};

}  // namespace TEST_NAMESPACE

#endif // SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_API_TYPES_FP64_H

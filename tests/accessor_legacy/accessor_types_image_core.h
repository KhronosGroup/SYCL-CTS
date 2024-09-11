/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provide common code for image accessor verification with core types
//
*******************************************************************************/

#ifndef SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_TYPES_IMAGE_CORE_H
#define SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_TYPES_IMAGE_CORE_H

#include "../common/common.h"
#include "../common/type_coverage.h"
#include "./../../util/extensions.h"

#ifndef TEST_NAME
#error Invalid test namespace
#endif

namespace TEST_NAMESPACE {

using user_alias = sycl::vec<sycl::opencl::cl_int, 4>;

/**
 *  @brief Run specific image accessors' tests for core type set
 */
template <template <typename, typename> class action,
          typename extensionTagT>
class check_all_types_image_core {

  template <typename T>
  using check_type = action<T, extensionTagT>;

public:
  static void run(sycl::queue& queue, sycl_cts::util::logger &log) {

    if (!queue.get_device().get_info<sycl::info::device::image_support>()) {
      log.note("Device does not support images -- skipping check");
      return;
    }

    // Skip tests in case extension not available
    using availability =
        sycl_cts::util::extensions::availability<extensionTagT>;
    if (!availability::check(queue, log))
      return;

    const auto types =
        named_type_pack<sycl::cl_int4,
                        sycl::cl_uint4,
                        sycl::cl_float4,
                        user_alias>::generate(
                        "sycl::opencl::cl_int",
                        "sycl::opencl::cl_uint",
                        "sycl::opencl::cl_float",
                        "user_alias");

    for_all_types<check_type>(types, log, queue);

    queue.wait_and_throw();
  }
};

}  // namespace TEST_NAMESPACE

#endif // SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_TYPES_IMAGE_CORE_H

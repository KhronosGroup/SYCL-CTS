/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Provide common code for image accessor verification with fp16 types
//
*******************************************************************************/

#ifndef SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_TYPES_IMAGE_FP16_H
#define SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_TYPES_IMAGE_FP16_H

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
class check_all_types_image_fp16 {

  using extension_tag_t = sycl_cts::util::extensions::tag::fp16;

  template <typename T>
  using check_type = action<T, extension_tag_t>;

public:
  static void run(cl::sycl::queue& queue, sycl_cts::util::logger &log) {

    if (!queue.get_device().get_info<cl::sycl::info::device::image_support>()) {
      log.note("Device does not support images -- skipping check");
      return;
    }

    // Skip tests in case extension is not available
    using availability =
        sycl_cts::util::extensions::availability<extension_tag_t>;
    if (!availability::check(queue, log))
      return;

    check_type<cl::sycl::cl_half4>()(log, queue, "cl::sycl::cl_half");

    queue.wait_and_throw();
  }
};

}  // namespace TEST_NAMESPACE

#endif // SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_TYPES_IMAGE_FP16_H

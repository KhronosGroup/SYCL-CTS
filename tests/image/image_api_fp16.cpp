/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides image get_access test for cl_half4
//
*******************************************************************************/

#include "../common/common.h"
#include "image_common.h"

#define TEST_NAME image_api_fp16

namespace TEST_NAMESPACE {

/**
 * test sycl::image initialization
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  template <int dims>
  void check_allocs(util::logger &log, image_api_check<dims> &img_check,
                    sycl::range<dims> &r, sycl::range<dims - 1> *p = nullptr) {
    img_check.template check_get_access<sycl::cl_half4, sycl::image_allocator>(
        log, sycl::image_channel_order::rgba, sycl::image_channel_type::fp16, r,
        p, true);
    img_check
        .template check_get_access<sycl::cl_half4, std::allocator<sycl::byte>>(
            log, sycl::image_channel_order::rgba,
            sycl::image_channel_type::fp16, r, p, true);
  }

  /** execute the test
   */
  void run(util::logger &log) override {
    auto queue = util::get_cts_object::queue();

    if (!queue.get_device().has(sycl::aspect::fp16)) {
      log.note(
          "Device does not support half precision floating point operations");
      return;
    }

    // Ensure the image always has 64 elements
    const int elemsPerDim1 = 64;
    const int elemsPerDim2 = 8;
    const int elemsPerDim3 = 4;

    sycl::range<1> range_1d(elemsPerDim1);
    sycl::range<2> range_2d(elemsPerDim2, elemsPerDim2);
    sycl::range<3> range_3d(elemsPerDim3, elemsPerDim3, elemsPerDim3);

    // Test without pitch
    {
      image_api_check<1> img_1d;
      image_api_check<2> img_2d;
      image_api_check<3> img_3d;
      check_allocs(log, img_1d, range_1d);
      check_allocs(log, img_2d, range_2d);
      check_allocs(log, img_3d, range_3d);
    }

    // Test with pitch
    {
      sycl::range<1> pitch_1d(elemsPerDim2);
      sycl::range<2> pitch_2d(elemsPerDim3, elemsPerDim3 * elemsPerDim3);

      image_api_check<2> img_2d;
      image_api_check<3> img_3d;
      check_allocs(log, img_2d, range_2d, &pitch_1d);
      check_allocs(log, img_3d, range_3d, &pitch_2d);
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */

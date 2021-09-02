/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"
#include "image_common.h"

#define TEST_NAME image_api_core

namespace TEST_NAMESPACE {
using namespace sycl_cts;

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

  /** execute the test
   */
  void run(util::logger &log) override {
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
      img_1d(log, range_1d);
      img_2d(log, range_2d);
      img_3d(log, range_3d);
    }

    // Test with pitch
    {
      sycl::range<1> pitch_1d(elemsPerDim2);
      sycl::range<2> pitch_2d(elemsPerDim3, elemsPerDim3 * elemsPerDim3);

      image_api_check<2> img_2d;
      image_api_check<3> img_3d;
      img_2d(log, range_2d, &pitch_1d);
      img_3d(log, range_3d, &pitch_2d);
    }

    // Check image enum classes
    {
      // Check image channel orders
      check_enum_class_value(sycl::image_channel_order::a);
      check_enum_class_value(sycl::image_channel_order::r);
      check_enum_class_value(sycl::image_channel_order::rx);
      check_enum_class_value(sycl::image_channel_order::rg);
      check_enum_class_value(sycl::image_channel_order::rgx);
      check_enum_class_value(sycl::image_channel_order::ra);
      check_enum_class_value(sycl::image_channel_order::rgb);
      check_enum_class_value(sycl::image_channel_order::rgbx);
      check_enum_class_value(sycl::image_channel_order::rgba);
      check_enum_class_value(sycl::image_channel_order::argb);
      check_enum_class_value(sycl::image_channel_order::bgra);
      check_enum_class_value(sycl::image_channel_order::intensity);
      check_enum_class_value(sycl::image_channel_order::luminance);
      check_enum_class_value(sycl::image_channel_order::abgr);

      // Check image channel types
      check_enum_class_value(sycl::image_channel_type::snorm_int8);
      check_enum_class_value(sycl::image_channel_type::snorm_int16);
      check_enum_class_value(sycl::image_channel_type::unorm_int8);
      check_enum_class_value(sycl::image_channel_type::unorm_int16);
      check_enum_class_value(sycl::image_channel_type::unorm_short_565);
      check_enum_class_value(sycl::image_channel_type::unorm_short_555);
      check_enum_class_value(sycl::image_channel_type::unorm_int_101010);
      check_enum_class_value(sycl::image_channel_type::signed_int8);
      check_enum_class_value(sycl::image_channel_type::signed_int16);
      check_enum_class_value(sycl::image_channel_type::signed_int32);
      check_enum_class_value(sycl::image_channel_type::unsigned_int8);
      check_enum_class_value(sycl::image_channel_type::unsigned_int16);
      check_enum_class_value(sycl::image_channel_type::unsigned_int32);
      check_enum_class_value(sycl::image_channel_type::fp16);
      check_enum_class_value(sycl::image_channel_type::fp32);
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */

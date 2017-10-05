/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME sampler_apis

namespace sampler_api__ {
using namespace sycl_cts;

/** tests the API for cl::sycl::sampler
*/
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute this test
  */
  void run(util::logger &log) override {
    try {
      // Ensure all addressing_mode values defined
      check_enum_class_value(cl::sycl::addressing_mode::mirrored_repeat);
      check_enum_class_value(cl::sycl::addressing_mode::repeat);
      check_enum_class_value(cl::sycl::addressing_mode::clamp_to_edge);
      check_enum_class_value(cl::sycl::addressing_mode::clamp);
      check_enum_class_value(cl::sycl::addressing_mode::none);
      check_enum_underlying_type<cl::sycl::addressing_mode, unsigned int>(log);

      // Ensure all filtering_mode values defined
      check_enum_class_value(cl::sycl::filtering_mode::nearest);
      check_enum_class_value(cl::sycl::filtering_mode::linear);
      check_enum_underlying_type<cl::sycl::filtering_mode, unsigned int>(log);

      // Ensure all coordinate_normalization_mode values defined
      check_enum_class_value(
          cl::sycl::coordinate_normalization_mode::normalized);
      check_enum_class_value(
          cl::sycl::coordinate_normalization_mode::unnormalized);
      check_enum_underlying_type<cl::sycl::coordinate_normalization_mode,
                                 unsigned int>(log);

      // Ensure all mipmap_filtering_mode values defined
      check_enum_class_value(cl::sycl::mipmap_filtering_mode::mipmap_none);
      check_enum_class_value(cl::sycl::mipmap_filtering_mode::mipmap_nearest);
      check_enum_class_value(cl::sycl::mipmap_filtering_mode::mipmap_linear);

      cl::sycl::sampler sampler(
          cl::sycl::coordinate_normalization_mode::unnormalized,
          cl::sycl::addressing_mode::none, cl::sycl::filtering_mode::nearest);

      /** check is_host() method
      */
      auto isHost = sampler.is_host();
      check_return_type<bool>(log, isHost, "is_host()");

      /** check get_addressing_mode() method
      */
      auto addressingMode = sampler.get_addressing_mode();
      check_return_type<cl::sycl::addressing_mode>(log, addressingMode,
                                                   "get_addressing_mode()");

      /** check get_filtering_mode() method
      */
      auto filterMode = sampler.get_filtering_mode();
      check_return_type<cl::sycl::filtering_mode>(log, filterMode,
                                                  "get_filtering_mode()");

      /** check get_coordinate_normalization_mode() method
      */
      auto isNormalizedCoordinates =
          sampler.get_coordinate_normalization_mode();
      check_return_type<cl::sycl::coordinate_normalization_mode>(
          log, isNormalizedCoordinates, "get_coordinate_normalization_mode()");

      auto queue = util::get_cts_object::queue();

      // Check the get() method
      if (!queue.is_host()) {
        cl::sycl::sampler sampler(
            cl::sycl::coordinate_normalization_mode::unnormalized,
            cl::sycl::addressing_mode::none, cl::sycl::filtering_mode::nearest);
        cl::sycl::image<2> inputImage(cl::sycl::image_channel_order::rgba,
                                      cl::sycl::image_channel_type::fp32,
                                      cl::sycl::range<2>(4, 1));
        cl::sycl::image<2> outputImage(cl::sycl::image_channel_order::rgba,
                                       cl::sycl::image_channel_type::fp32,
                                       cl::sycl::range<2>(4, 1));

        queue.submit([&](cl::sycl::handler &handler) {
          auto inPtr =
              inputImage
                  .get_access<cl::sycl::float4, cl::sycl::access::mode::read>(
                      handler);
          auto outPtr =
              outputImage
                  .get_access<cl::sycl::float4, cl::sycl::access::mode::write>(
                      handler);

          // Each command group requires exactly one kernel
          // Ensures outerSampler is used within a context
          handler.parallel_for<class TEST_NAME>(
              cl::sycl::nd_range<2>(cl::sycl::range<2>(4, 1),
                                    cl::sycl::range<2>(1, 1)),
              ([=](cl::sycl::nd_item<2> itemID) {
                outPtr[itemID.get_global()] =
                    inPtr(sampler)[itemID.get_global()];
              }));
        });

        queue.wait_and_throw();

        auto clSampler = sampler.get();
        check_return_type<cl_sampler>(log, clSampler,
                                      "cl::sycl::sampler::get()");
        if (clSampler == nullptr) {
          FAIL(log, "Retrieved null OpenCL sampler (get)");
        }
      }
    } catch (const cl::sycl::exception &e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace sampler_api__ */

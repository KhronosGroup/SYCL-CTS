/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME sampler_apis

namespace sampler_api__ {
using namespace sycl_cts;

/** tests the API for sycl::sampler
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
      check_enum_class_value(sycl::addressing_mode::mirrored_repeat);
      check_enum_class_value(sycl::addressing_mode::repeat);
      check_enum_class_value(sycl::addressing_mode::clamp_to_edge);
      check_enum_class_value(sycl::addressing_mode::clamp);
      check_enum_class_value(sycl::addressing_mode::none);
      check_enum_underlying_type<sycl::addressing_mode, unsigned int>(log);

      // Ensure all filtering_mode values defined
      check_enum_class_value(sycl::filtering_mode::nearest);
      check_enum_class_value(sycl::filtering_mode::linear);
      check_enum_underlying_type<sycl::filtering_mode, unsigned int>(log);

      // Ensure all coordinate_normalization_mode values defined
      check_enum_class_value(
          sycl::coordinate_normalization_mode::normalized);
      check_enum_class_value(
          sycl::coordinate_normalization_mode::unnormalized);
      check_enum_underlying_type<sycl::coordinate_normalization_mode,
                                 unsigned int>(log);

      sycl::sampler sampler(
          sycl::coordinate_normalization_mode::unnormalized,
          sycl::addressing_mode::none, sycl::filtering_mode::nearest);

      /** check get_addressing_mode() method
      */
      auto addressingMode = sampler.get_addressing_mode();
      check_return_type<sycl::addressing_mode>(log, addressingMode,
                                                   "get_addressing_mode()");

      /** check get_filtering_mode() method
      */
      auto filterMode = sampler.get_filtering_mode();
      check_return_type<sycl::filtering_mode>(log, filterMode,
                                                  "get_filtering_mode()");

    } catch (const sycl::exception &e) {
      log_exception(log, e);
      sycl::string_class errorMsg =
          "a SYCL exception was caught: " + sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace sampler_api__ */

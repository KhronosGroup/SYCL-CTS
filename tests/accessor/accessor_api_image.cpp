/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"
#include "./../../util/math_helper.h"
#include "accessor_api_image_common.h"
#include "accessor_utility.h"

#include <array>
#include <numeric>
#include <sstream>

#define TEST_NAME accessor_api_image

namespace TEST_NAMESPACE {

using namespace sycl_cts;
using namespace accessor_utility;

/** tests the api for cl::sycl::accessor
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
      auto queue = util::get_cts_object::queue();
      if (!queue.get_device()
               .get_info<cl::sycl::info::device::image_support>()) {
        log.note("Device does not support images -- skipping check");
        return;
      }

      /** check image accessor api for cl_int4
       */
      check_image_accessor_api_type<cl::sycl::cl_int4>(log, queue);

      /** check image accessor api for cl_uint4
       */
      check_image_accessor_api_type<cl::sycl::cl_uint4>(log, queue);

      /** check image accessor api for cl_float4
       */
      check_image_accessor_api_type<cl::sycl::cl_float4>(log, queue);

      queue.wait_and_throw();
    } catch (const cl::sycl::exception &e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

/** register this test with the test_collection
*/
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE

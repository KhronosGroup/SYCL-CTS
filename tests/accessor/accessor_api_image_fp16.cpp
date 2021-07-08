/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#define TEST_NAME accessor_api_image_fp16

#include "../common/common.h"
#include "./../../util/math_helper.h"
#include "accessor_api_image_common.h"
#include "accessor_types_image_fp16.h"

#include <array>
#include <numeric>
#include <sstream>

namespace TEST_NAMESPACE {

using namespace sycl_cts;
using namespace accessor_utility;

/** tests the api for sycl::accessor
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

      check_all_types_image_fp16<check_image_accessor_api_type>::run(
          queue, log);

      queue.wait_and_throw();
    } catch (const sycl::exception &e) {
      log_exception(log, e);
      std::string errorMsg =
          "a SYCL exception was caught: " + std::string(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

/** register this test with the test_collection
*/
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE

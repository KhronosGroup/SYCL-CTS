/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#define TEST_NAME accessor_api_buffer_fp64

#include "../common/common.h"
#include "accessor_api_buffer_common.h"
#include "accessor_types_fp64.h"

namespace TEST_NAMESPACE {

using namespace sycl_cts;

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
    auto queue = util::get_cts_object::queue();

    check_all_types_fp64<check_buffer_accessor_api_type>::run(queue, log);

    queue.wait_and_throw();
  }
};

/** register this test with the test_collection
 */
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE

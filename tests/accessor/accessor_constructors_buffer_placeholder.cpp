/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#define TEST_NAME accessor_constructors_buffer_placeholder

#include "../common/common.h"
#include "accessor_constructors_buffer_utility.h"
#include "accessor_types_core.h"

namespace TEST_NAMESPACE {

using namespace accessor_utility;

/** tests the constructors for sycl::accessor
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

    using extension_tag = sycl_cts::util::extensions::tag::core;

    check_all_types_core<buffer_accessor_type_placeholder, extension_tag>::run(
        queue, log);

    queue.wait_and_throw();
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE

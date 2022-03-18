/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
//  Provides tests for legacy multi_ptr constructors
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/type_list.h"
#include "multi_ptr_legacy_constructors_common.h"

#include <string>

#define TEST_NAME multi_ptr_legacy_constructors_core

namespace TEST_NAMESPACE {
using namespace multi_ptr_legacy_constructors_common;
using namespace sycl_cts;

/** tests the constructors for explicit pointers
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
#ifdef SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
    auto queue = util::get_cts_object::queue();

    const auto types = get_types();

    for_all_types<check_void_pointer_ctors>(types, queue);
    for_all_types<check_pointer_ctors>(types, queue);

    check_pointer_ctors<user_struct>{}(queue, "user_struct");

    queue.wait_and_throw();
#else
    log.note(
        "The test is skipped because tests for the deprecated features are "
        "disabled.");
#endif  // SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */

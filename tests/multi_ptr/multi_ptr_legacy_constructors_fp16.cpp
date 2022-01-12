/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for multi_ptr constructors with fp16 data types.
//
*******************************************************************************/

#include "../common/common.h"
#include "multi_ptr_legacy_constructors_common.h"

#include <string>

#define TEST_NAME multi_ptr_legacy_constructors_fp16

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
#ifdef SYCL_CTS_TEST_DEPRECATED_FEATURES
    auto queue = util::get_cts_object::queue();

    if (!queue.get_device().has(sycl::aspect::fp16)) {
      log.note(
          "Device does not support half precision floating point operations");
      return;
    }

    check_void_pointer_ctors<sycl::half>{}(queue, "sycl::half");
    check_pointer_ctors<sycl::half>{}(queue, "sycl::half");

    queue.wait_and_throw();
#else
    log.note(
        "The test is skipped because tests for the deprecated features are "
        "disabled.");
#endif  // SYCL_CTS_TEST_DEPRECATED_FEATURES
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */

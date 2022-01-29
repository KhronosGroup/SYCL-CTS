/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for multi_ptr members tests with sycl::half data type.
//
*******************************************************************************/

#include "../../util/extensions.h"
#include "../common/common.h"
#include "multi_ptr_members.h"

#define TEST_NAME multi_ptr_members_fp16

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace multi_ptr_members;

/** tests the api for explicit pointers
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
    using avaliability =
        util::extensions::availability<util::extensions::tag::fp16>;
    if (!avaliability::check(queue, log)) {
      WARN("Device does not support half precision floating point operations");
      return;
    }

    run_test_with_chosen_data_type<sycl::half>{}(log, "sycl::half");
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */

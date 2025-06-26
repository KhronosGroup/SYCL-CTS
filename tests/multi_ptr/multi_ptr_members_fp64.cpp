/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for multi_ptr members tests with double data type.
//
*******************************************************************************/

#include "../common/common.h"
#include "multi_ptr_members.h"

#define TEST_NAME multi_ptr_members_fp64

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
    if (!queue.get_device().has(sycl::aspect::fp64)) {
      WARN(
          "Device does not support double precision floating point operations "
          "- skipping the test");
      return;
    }

    run_test_with_chosen_data_type<double>{}(log, "double");
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */

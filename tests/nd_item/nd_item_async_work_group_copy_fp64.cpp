/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides nd_item async_work_group_copy tests for double and cl_double
//
*******************************************************************************/

#include "nd_item_async_work_group_copy_common.h"

#define TEST_NAME nd_item_async_work_group_copy_fp64

using namespace sycl_cts;
using namespace nd_item_async_work_group_copy;

namespace TEST_NAMESPACE {

class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  void run(util::logger &log) override {
    auto queue = util::get_cts_object::queue();

    if (!queue.get_device().has(sycl::aspect::fp64)) {
      log.note(
          "Device does not support double precision floating point operations");
      return;
    }
    // Test using queue constructed already
    for_type_and_vectors<check_type, double>(queue, log, "double");
    for_type_and_vectors<check_type, sycl::cl_double>(queue, log,
                                                      "sycl::cl_double");
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE

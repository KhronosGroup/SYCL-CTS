/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Provides group async_work_group_copy tests for double and cl_double
//
*******************************************************************************/

#include "group_async_work_group_copy_common.h"

#define TEST_NAME group_async_work_group_copy_fp64

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

    if (!queue.get_device().has_extension("cl_khr_fp64")) {
      log.note(
        "Device does not support double precision floating point operations");
      return;
    }

    check_type_and_vec<double>(log);
    check_type_and_vec<cl::sycl::cl_double>(log);
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE

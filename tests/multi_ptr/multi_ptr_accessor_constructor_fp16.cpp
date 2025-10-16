/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for multi_ptr accessor constructor for sycl::half
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/type_coverage.h"
#include "multi_ptr_accessor_constructor.h"

#define TEST_NAME multi_ptr_accessor_constructor_fp16

namespace TEST_NAMESPACE {
using namespace sycl_cts;

/** test multi_ptr accessor constructor
 */
class TEST_NAME : public sycl_cts::util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  void run(util::logger &log) override {
    using namespace multi_ptr_accessor_constructors;

    auto queue = util::get_cts_object::queue();

    if (!queue.get_device().has(sycl::aspect::fp16)) {
      SKIP("Device does not support half precision floating point operations");
    }

    check_multi_ptr_accessor_constructor_for_type<sycl::half>{}(log,
                                                                "sycl::half");
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE

/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for multi_ptr common constructors for sycl::half
//
*******************************************************************************/

#include "../../util/extensions.h"
#include "../common/common.h"
#include "../common/type_coverage.h"

#include "multi_ptr_common_constructors.h"

#define TEST_NAME multi_ptr_common_constructors_fp16

namespace TEST_NAMESPACE {
using namespace sycl_cts;

/** test multi_ptr common constructors with sycl::half
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
    using namespace multi_ptr_common_constructors;

    auto queue = util::get_cts_object::queue();
    using avaliability =
        util::extensions::availability<util::extensions::tag::fp16>;
    if (!avaliability::check(queue, log)) {
      WARN("Device does not support half precision floating point operations");
      return;
    }

    check_multi_ptr_common_constructors_for_type<sycl::half>{}(log,
                                                               "sycl::half");
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE

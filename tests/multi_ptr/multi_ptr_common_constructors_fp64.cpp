/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for multi_ptr common constructors for double
//
*******************************************************************************/

#include "../../util/extensions.h"
#include "../common/common.h"
#include "../common/type_coverage.h"

#include "multi_ptr_common_constructors.h"

#define TEST_NAME multi_ptr_common_constructors_fp64

namespace TEST_NAMESPACE {
using namespace sycl_cts;

/** test multi_ptr common constructors with double
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
        util::extensions::availability<util::extensions::tag::fp64>;
    if (!avaliability::check(queue, log)) {
      log.skip(
          "Device does not support double precision floating point operations");
      return;
    }

    check_multi_ptr_common_constructors_for_type<double>{}(log, "double");
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE

/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for sycl::compile to check associated devices.
//  Overload (bundle, pl)
//
*******************************************************************************/

#include "../common/common.h"
#include "sycl_compile.h"

#define TEST_NAME sycl_compile_check_associated_devices_bundle_only

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace sycl_cts::tests::sycl_compile;

/** test sycl::compile to check associated devices
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
    constexpr auto overload = CompileOverload::bundle_only;
    check_associated_devices<overload>(log);
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE

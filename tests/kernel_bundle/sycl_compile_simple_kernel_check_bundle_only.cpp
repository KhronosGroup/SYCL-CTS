/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for sycl::compile to check that all kernels without optional
//  features are represented in result bundle. Overload (bundle, devices, pl)
//
*******************************************************************************/

#include "../common/common.h"
#include "sycl_compile.h"
#include "kernels.h"

#include <algorithm>

#define TEST_NAME sycl_compile_simple_kernel_check_bundle_only

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace sycl_cts::tests::sycl_compile;

using simple_descr = kernels::simple_kernel_descriptor;

/** test sycl::compile to check all kernels without opt. features existence in
 *  result bundle
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
    check_bundle_kernels<simple_descr, overload>(log, "simple_kernel");
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE

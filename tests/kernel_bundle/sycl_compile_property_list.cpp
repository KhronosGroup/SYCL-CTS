/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for sycl::compile where property_list is passed or omitted.
//  Overload (bundle, devices, pl)
//
*******************************************************************************/

#include "../common/common.h"
#include "sycl_compile.h"
#include "kernel_bundle.h"

#define TEST_NAME sycl_compile_property_list

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace sycl_cts::tests::sycl_compile;

/** test sycl::compile where property_list is passed or omitted
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
    auto queue = util::get_cts_object::queue();
    auto ctx = queue.get_context();

    if ( !sycl::has_kernel_bundle<sycl::bundle_state::input>(ctx) ) {
      log.note("No kernel bundle with input state for test (skipped).");
      return;
    }

    auto input_kb = sycl::get_kernel_bundle<sycl::bundle_state::input>(ctx);

    // Call sycl::compile without passing property list.
    // Overload (bundle, pl - not passed)
    {
      sycl::compile(input_kb);
    }

    // Call sycl::compile with passing empty property list
    // Overload (bundle, pl)
    {
      sycl::compile(input_kb, sycl::property_list{});
    }

    // Call sycl::compile without passing property list
    // Overload (bundle, devices, pl - not passed)
    {
      sycl::compile(input_kb, ctx.get_devices());
    }

    // Call sycl::compile with passing empty property list
    // Overload (bundle, devices, pl)
    {
      sycl::compile(input_kb, ctx.get_devices(), {});
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE

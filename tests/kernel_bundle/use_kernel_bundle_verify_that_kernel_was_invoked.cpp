/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  This test obtain kernel bundle in bundle_state::input, then for this kernel
//  calls set_spec_const with some different value, then call sycl::build to
//  have kernel bundle in bundle_state::executable state and calls
//  sycl::handler::use_kernel_bundle for this bundle, then get spec const value.
//
//  The test verifies that the specialization constant value is equal to
//  expected value.
//
//  This test should pass even if implementation of use_kernel_bundle is empty.
//
*******************************************************************************/

#include "../common/common.h"
#include "use_kernel_bundle.h"

#define TEST_NAME use_kernel_bundle_verify_that_kernel_was_invoked

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace sycl_cts::tests::use_kernel_bundle;

constexpr int spec_const_def_value{10};
constexpr sycl::specialization_id<int> spec_const{spec_const_def_value};

constexpr int spec_const_non_def_value{20};
constexpr int expected_spec_const_non_value{spec_const_non_def_value};

struct kernel;

/** @brief Call kernel from bundle and verify that this kernel was invoked
 *  @param log sycl_cts::util::logger class object
 *  @param ctx Context that will used for sycl::queue
 */
void invoke_kernel_and_verify_invocation(util::logger &log,
                                         const sycl::context &ctx) {
  sycl::queue queue(ctx, ctx.get_devices()[0]);
  bool flags[] = {false, false};

  {
    sycl::buffer<bool> flag_buffer{flags, sycl::range<1>{2}};
    queue.submit([&](sycl::handler &cgh) {
      auto kernel_bundle{
          sycl::get_kernel_bundle<kernel, sycl::bundle_state::input>(ctx)};

      kernel_bundle.set_specialization_constant<spec_const>(
          spec_const_non_def_value);

      auto exec_bundle{sycl::build(kernel_bundle)};
      cgh.use_kernel_bundle(exec_bundle);

      auto accessor_to_flag{
          sycl::accessor<bool, 1, sycl::access_mode::write,
                         sycl::target::global_buffer>(flag_buffer, cgh)};

      cgh.single_task<kernel>([=](sycl::kernel_handler h) {
        accessor_to_flag[0] = h.get_specialization_constant<spec_const>() ==
                              expected_spec_const_non_value;
        accessor_to_flag[1] = std::is_same_v<
            typename std::remove_reference_t<decltype(spec_const)>::value_type,
            decltype(h.get_specialization_constant<spec_const>())>;
      });
    });
  }

  {
    INFO(
        "The specialization constant value not equal to expected "
        "specialization constant value");
    CHECK(flags[0]);
  }
  {
    INFO(
        "Check kernel_handler::get_specialization_constant() "
        "return type");
    CHECK(flags[1]);
  }
}

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
    sycl::device dev = util::get_cts_object::device();
    sycl::context ctx(dev.get_platform().get_devices());

    invoke_kernel_and_verify_invocation(log, ctx);
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE

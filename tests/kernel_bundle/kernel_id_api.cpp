/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for sycl::kernel_id API
//
*******************************************************************************/

#include "../common/common.h"
#include "get_kernel_id.h"
#include "kernel_bundle.h"

#include <type_traits>

#define TEST_NAME kernel_id_api

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace sycl_cts::tests::get_kernel_id;
using namespace sycl_cts::tests::kernel_bundle;

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
    /** Check that kernel_id does not have default constructor
     */
    if (std::is_default_constructible_v<sycl::kernel_id>) {
      FAIL(log, "sycl::kernel_id should not have default constructor");
    }

    /** Check that kernel_id is destructible
     */
    if (!std::is_destructible_v<sycl::kernel_id>) {
      FAIL(log, "sycl::kernel_id should be destructible");
    }

    /** Check kernel_id::get_name() return type
     */
    using RetType = std::invoke_result_t<decltype(&sycl::kernel_id::get_name),
                                         sycl::kernel_id>;
    if (!std::is_same_v<RetType, const char *>) {
      FAIL(log, "Incorrect return type of kernel_id::get_name()");
    }

    /** Check const correctness of kernel_id::get_name()
     */
    run_verification<by_handler>(log, [&](sycl::handler &cgh) {
      using kernel_name = class k_id_get_name_const_correctness;
      const sycl::kernel_id k_id = sycl::get_kernel_id<kernel_name>();
      if (k_id.get_name() == nullptr) {
        FAIL(log, "sycl::kernel_id::get_name returned nullptr.");
      }

      cgh.single_task<kernel_name>([=] {});
    });

    /** Check noexcept specifier for kernel_id::get_name()
     */
    run_verification<by_handler>(log, [&](sycl::handler &cgh) {
      if constexpr (!noexcept(std::declval<sycl::kernel_id>().get_name())) {
        FAIL(log, "Missing 'noexcept' specifier for kernel_id::get_name()");
      }
    });
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE

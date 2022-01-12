/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for sycl::kernel_id::get_kernel_id()
//
*******************************************************************************/

#include "../common/common.h"
#include "get_kernel_id.h"
#include "kernel_bundle.h"

#define TEST_NAME get_kernel_id_function

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
    /** Check return type of sycl::get_kernel_id()
     */
    run_verification<by_queue>(log, [&](sycl::queue &queue) {
      using kernel_name = class k_id_ret_type;
      using RetType =
          std::invoke_result_t<decltype(sycl::get_kernel_id<k_id_ret_type>)>;
      if (!std::is_same_v<RetType, sycl::kernel_id>) {
        FAIL(log, "Wrong return type of sycl::get_kernel_id()");
      }

      define_kernel<kernel_name>(queue);
    });

    /** Call sycl::get_kernel_id() for two existing kernels and make
     *  sure that return is different
     */
    run_verification<by_queue>(log, [&](sycl::queue &queue) {
      using kernel_name = class k_id_get_kernel_id;
      using kernel_name_other = class k_id_get_kernel_id_other;
      sycl::kernel_id k_id = sycl::get_kernel_id<kernel_name>();
      sycl::kernel_id k_id_other = sycl::get_kernel_id<kernel_name_other>();

      if (k_id == k_id_other) {
        FAIL(log, "Expected different kernel_ids for different kernels");
      }
      define_kernel<kernel_name>(queue);
      define_kernel<kernel_name_other>(queue);
    });
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE

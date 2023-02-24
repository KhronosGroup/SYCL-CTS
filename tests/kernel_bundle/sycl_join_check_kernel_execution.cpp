/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests to check kernel execution after sycl::join
//  (State == executable only)
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/get_cts_string.h"
#include "sycl_join.h"

#include <algorithm>

#define TEST_NAME sycl_join_check_kernel_execution

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace sycl_cts::tests::sycl_join;

struct kb_join_post_exe_kernel;

/** @brief Execute given kernel and check that the kernel produces correct
 *         results
 *  @tparam KernelName Name of kernel to check
 */
template <typename KernelName>
void run_verification(util::logger &log, sycl::queue &q,
                      sycl::kernel_bundle<sycl::bundle_state::executable> &kb) {
  bool value_to_rewrite_in_kernel = false;
  {
    sycl::buffer<bool, 1> res_buffer(&value_to_rewrite_in_kernel,
                                     sycl::range<1>(1));
    q.submit([&](sycl::handler &cgh) {
      auto res_acc =
          res_buffer.template get_access<sycl::access_mode::write>(cgh);
      cgh.use_kernel_bundle(kb);
      cgh.single_task<KernelName>([=] { res_acc[0] = true; });
    });
  }
  if (!value_to_rewrite_in_kernel) {
    FAIL(log, "Kernel results are not expected");
  }
}

/** test kernel execution after sycl::join
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
    using k_name = kb_join_post_exe_kernel;
    auto queue = util::get_cts_object::queue();
    const auto ctx = queue.get_context();
    const auto dev = queue.get_device();

    auto kb_1 =
        sycl::get_kernel_bundle<sycl::bundle_state::executable>(ctx, {dev});
    auto kb_2 =
        sycl::get_kernel_bundle<sycl::bundle_state::executable>(ctx, {dev});
    auto k_id = sycl::get_kernel_id<k_name>();

    auto joined_kb = sycl::join<sycl::bundle_state::executable>({kb_1, kb_2});
    run_verification<k_name>(log, queue, joined_kb);
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE

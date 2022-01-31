/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for sycl::kernel_id::get_kernel_ids() for multiple kernels
//  in application.
//
//  IMPORTANT: This file should not be merged with another *.cpp file because we
//  must have a strictly defined kernels in the application because tested
//  function uses all available kernels and we can't chose specific kernel.
//
*******************************************************************************/

#include "../common/common.h"
#include "get_kernel_id.h"
#include "kernel_bundle.h"

#define TEST_NAME get_kernel_ids_multiple_kernels

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace sycl_cts::tests::get_kernel_id;
using namespace sycl_cts::tests::kernel_bundle;

template <size_t idx>
class get_kernel_ids_multiple;

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
    /** Check return type of sycl::get_kernel_ids()
     */
    {
      using RetType = std::invoke_result_t<decltype(sycl::get_kernel_ids)>;
      if (!std::is_same_v<RetType, std::vector<sycl::kernel_id>>) {
        FAIL(log, "Wrong return type of sycl::get_kernel_id()");
      }
    }

    /** Call sycl::get_kernel_id() for multiple kernels in application
     */
    run_verification<by_queue>(log, [&](sycl::queue &queue) {
      constexpr size_t kernels_count = 4;
      using k_name_1 = get_kernel_ids_multiple<1>;
      using k_name_2 = get_kernel_ids_multiple<2>;
      using k_name_3 = get_kernel_ids_multiple<3>;
      using k_name_4 = get_kernel_ids_multiple<4>;
      auto k_ids = sycl::get_kernel_ids();
      if (k_ids.size() == 0) {
        FAIL(log,
             "Empty kernel_ids vector returned. Vector should include"
             "exactly " +
                 std::to_string(kernels_count) + " kernel_id");
      } else if (k_ids.size() < kernels_count) {
        FAIL(log,
             "Less than expected kernel_ids in returned vector."
             " Expected: " +
                 std::to_string(kernels_count) +
                 " Returned: " + std::to_string(k_ids.size()));
      } else if (k_ids.size() > kernels_count) {
        FAIL(log,
             "More than expected kernel_ids in returned vector."
             " Expected: " +
                 std::to_string(kernels_count) +
                 " Returned: " + std::to_string(k_ids.size()));
      }
      // Check that all kernel_ids are different
      for (auto it_main = k_ids.begin(); it_main != k_ids.end(); ++it_main) {
        for (auto it_nes = it_main + 1; it_nes != k_ids.end(); ++it_nes) {
          if (*it_main == *it_nes)
            FAIL(log,
                 "Multiple kernels in application should provide non-equal"
                 "kernel_ids.");
        }
      }
      define_kernel<k_name_1>(queue);
      define_kernel<k_name_2>(queue);
      define_kernel<k_name_3>(queue);
      define_kernel<k_name_4>(queue);
    });
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE

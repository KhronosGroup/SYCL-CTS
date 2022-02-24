/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for the exception that are thrown when device
//  doesn’t support required work group size.
//
*******************************************************************************/

#include "../common/common.h"
#include "catch2/catch_template_test_macros.hpp"
#include "kernel_features_common.h"

namespace kernel_features_reqd_work_group_size {
using namespace sycl_cts;
using namespace kernel_features_common;

template <size_t N>
class Functor {
 public:
  [[sycl::reqd_work_group_size(N)]] void operator()() const {}
  [[sycl::reqd_work_group_size(N)]] void operator()(sycl::item<1>) const {}
  [[sycl::reqd_work_group_size(N)]] void operator()(sycl::group<1>) const {}
};

TEMPLATE_TEST_CASE_SIG(
    "Kernel features. Exceptions thrown by [[reqd_work_group_size(N)]] with "
    "unsupported size",
    "[kernel_features]", ((size_t N), N), 16, 4294967295) {
  auto queue = util::get_cts_object::queue();
  auto max_wg_size =
      queue.get_device().get_info<sycl::info::device::max_work_group_size>();

  bool is_exception_expected = (N > max_wg_size);

  // Set expected error code
  constexpr sycl::errc expected_errc = sycl::errc::kernel_not_supported;

  {
    const auto lambda_no_arg = []() [[sycl::reqd_work_group_size(N)]]{};
    const auto lambda_item_arg =
        [](sycl::item<1>) [[sycl::reqd_work_group_size(N)]]{};
    const auto lambda_group_arg =
        [](sycl::group<1>) [[sycl::reqd_work_group_size(N)]]{};

    run_separate_lambda(is_exception_expected, expected_errc, queue,
                        lambda_no_arg, lambda_item_arg, lambda_group_arg);
  }
  { run_functor<Functor<N>>(is_exception_expected, expected_errc, queue); }
  {
    RUN_SUBMISSION_CALL(is_exception_expected, expected_errc, queue,
                        [[sycl::reqd_work_group_size(N)]], NO_KERNEL_BODY);
  }
}

}  // namespace kernel_features_reqd_work_group_size

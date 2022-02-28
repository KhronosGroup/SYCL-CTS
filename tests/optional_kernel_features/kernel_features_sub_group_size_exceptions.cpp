/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for the exception that are thrown by
//  [[sycl::reqd_sub_group_size(N)]] attribute with unsupported N.
//
*******************************************************************************/

#include "catch2/catch_template_test_macros.hpp"
#include "kernel_features_common.h"
#include <algorithm>

namespace kernel_features_sub_group_size {
using namespace sycl_cts;
using namespace kernel_features_common;

template <size_t N>
class functor_with_attribute {
 public:
  [[sycl::reqd_sub_group_size(N)]] void operator()() const {}
  [[sycl::reqd_sub_group_size(N)]] void operator()(sycl::item<1>) const {}
  [[sycl::reqd_sub_group_size(N)]] void operator()(sycl::group<1>) const {}
};

TEMPLATE_TEST_CASE_SIG(
    "Runtime exception if device doesn't support required sub-group size",
    "[kernel_features]", ((size_t N), N), 16, 4099) {
  auto queue = util::get_cts_object::queue();

  const sycl::errc errc_expected = sycl::errc::kernel_not_supported;
  bool is_exception_expected = true;

  // Verify N supported or not supported as sub_group size on the current device
  const auto sg_sizes_vec =
      queue.get_device().get_info<sycl::info::device::sub_group_sizes>();
  auto find_res = std::find(sg_sizes_vec.begin(), sg_sizes_vec.end(), N);
  if (find_res != sg_sizes_vec.end()) {
    is_exception_expected = false;
  }

  {
    const auto separate_lambda_no_arg = []() [[sycl::reqd_sub_group_size(N)]]{};
    const auto separate_lambda_item_arg =
        [](sycl::item<1>) [[sycl::reqd_sub_group_size(N)]]{};
    const auto separate_lambda_group_arg =
        [](sycl::group<1>) [[sycl::reqd_sub_group_size(N)]]{};

    run_separate_lambda(is_exception_expected, errc_expected, queue,
                        separate_lambda_no_arg, separate_lambda_item_arg,
                        separate_lambda_group_arg);
  }

  {
    using FunctorT = functor_with_attribute<N>;
    run_functor<FunctorT>(is_exception_expected, errc_expected, queue);
  }

  {
    RUN_SUBMISSION_CALL(is_exception_expected, errc_expected, queue,
                        [[sycl::reqd_sub_group_size(N)]], {});
  }
}
}  // namespace kernel_features_sub_group_size

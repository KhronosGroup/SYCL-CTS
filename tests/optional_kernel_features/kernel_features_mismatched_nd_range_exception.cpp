/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for the exception that is thrown by
//  [[sycl::reqd_work_group_size]] attribute when use nd_range of wrong size in
//  kernel.
//
*******************************************************************************/

#include "../common/common.h"
#include "catch2/catch_template_test_macros.hpp"

namespace kernel_features_mismatched_nd_range_exception {
using namespace sycl_cts;

struct kernel_separate_lambda;
struct kernel_functor;
struct kernel_submission_call;

// Define required size of work group for attribute
constexpr int testing_wg_size = 16;

class Functor {
 public:
  [[sycl::reqd_work_group_size(testing_wg_size)]] void operator()(
      sycl::nd_item<1>) const {}
};

TEST_CASE(
    "Kernel features. Exceptions throwns by [[reqd_work_group_size()]] with "
    "mismatched nd_range",
    "[kernel_features]") {
  auto queue = util::get_cts_object::queue();
  const size_t max_wg_size =
      queue.get_device().get_info<sycl::info::device::max_work_group_size>();
  if (max_wg_size < testing_wg_size) {
    WARN("Device supported work group size too small. Skipping..");
    return;
  }

  const bool is_exception_expected = true;
  sycl::errc errc_expected = sycl::errc::nd_range;

  const auto separate_lambda = [=](sycl::nd_item<1>)
      [[sycl::reqd_work_group_size(testing_wg_size)]]{};

  // Create nd_range that have to cause an exception
  sycl::nd_range<1> mismatched_nd_rage(sycl::range(max_wg_size + 1),
                                       sycl::range(max_wg_size + 1));

  {
    INFO("Task as separate lambda");
    bool is_exception_thrown = false;
    try {
      queue
          .submit([&](sycl::handler& cgh) {
            cgh.parallel_for<kernel_separate_lambda>(mismatched_nd_rage,
                                                     separate_lambda);
          })
          .wait_and_throw();
    } catch (const sycl::exception& e) {
      is_exception_thrown = true;
      INFO("Error code check");
      CHECK(e.code() == errc_expected);
    }
    CHECK(is_exception_expected == is_exception_thrown);
  }

  {
    INFO("Task as functor");
    bool is_exception_thrown = false;
    try {
      queue
          .submit([&](sycl::handler& cgh) {
            cgh.parallel_for<kernel_functor>(mismatched_nd_rage, Functor{});
          })
          .wait_and_throw();
    } catch (const sycl::exception& e) {
      is_exception_thrown = true;
      INFO("Error code check");
      CHECK(e.code() == errc_expected);
    }
    CHECK(is_exception_expected == is_exception_thrown);
  }

  {
    INFO("Task as submission call");
    bool is_exception_thrown = false;
    try {
      queue
          .submit([&](sycl::handler& cgh) {
            cgh.parallel_for<kernel_submission_call>(
                mismatched_nd_rage,
                [](sycl::nd_item<1>)
                    [[sycl::reqd_work_group_size(testing_wg_size)]]{});
          })
          .wait_and_throw();
    } catch (const sycl::exception& e) {
      is_exception_thrown = true;
      INFO("Error code check");
      CHECK(e.code() == errc_expected);
    }
    CHECK(is_exception_expected == is_exception_thrown);
  }
}
}  // namespace kernel_features_mismatched_nd_range_exception

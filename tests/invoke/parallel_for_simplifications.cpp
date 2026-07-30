/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for parallel_for simplifications
//  parallel_for(N, some_kernel)
//  parallel_for({N}, some_kernel)
//  parallel_for({N1, N2}, some_kernel)
//  parallel_for({N1, N2, N3}, some_kernel)
//
//  Test plan: /test_plans/parallel_for_simplifications.asciidoc
//
*******************************************************************************/

#include "../common/disabled_for_test_case.h"
#include "catch2/catch_template_test_macros.hpp"

#include "../common/get_cts_object.h"

namespace parallel_for_simplifications {

template <int dim>
struct kernel {
  using accessor_t =
      sycl::accessor<int, 1, sycl::access_mode::write, sycl::target::device>;
  accessor_t m_acc;

  kernel(accessor_t acc) : m_acc(acc) {}

  void operator()(sycl::item<dim> item) const {
    size_t index = item.get_linear_id();
    m_acc[index] = index;
  }
};

template <int N, typename ActionT>
void check(ActionT action) {
  std::array<int, N> arr{};
  {
    sycl::buffer<int, 1> buf(arr.data(), sycl::range<1>(N));
    sycl::queue queue = sycl_cts::util::get_cts_object::queue();

    queue.submit([&](sycl::handler& cgh) {
      auto acc = buf.get_access<sycl::access_mode::write>(cgh);
      action(cgh, acc);
    });
  }
  for (int i = 0; i < N; i++) CHECK(arr[i] == i);
}

// FIXME: re-enable when parallel_for simplifications implemented in AdaptiveCpp
DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("Check parallel_for(N, some_kernel)", "[parallel_for_simplifications]")({
  constexpr int N = 2;
  auto action = [=](auto& cgh, auto& acc) {
    cgh.parallel_for(N, kernel<1>(acc));
  };
  check<N>(action);
});

// FIXME: re-enable when parallel_for simplifications implemented in AdaptiveCpp
DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("Check parallel_for({N}, some_kernel)", "[parallel_for_simplifications]")({
  constexpr int N = 2;
  auto action = [=](auto& cgh, auto& acc) {
    cgh.parallel_for({N}, kernel<1>(acc));
  };
  check<N>(action);
});

// FIXME: re-enable when parallel_for simplifications implemented in AdaptiveCpp
// / SimSYCL
DISABLED_FOR_TEST_CASE(AdaptiveCpp, SimSYCL)
("Check parallel_for({N1, N2}, some_kernel)",
 "[parallel_for_simplifications]")({
  constexpr int N1 = 2;
  constexpr int N2 = 3;
  constexpr int N = N1 * N2;
  auto action = [=](auto& cgh, auto& acc) {
    cgh.parallel_for({N1, N2}, kernel<2>(acc));
  };
  check<N>(action);
});

// FIXME: re-enable when parallel_for simplifications implemented in AdaptiveCpp
// / SimSYCL
DISABLED_FOR_TEST_CASE(AdaptiveCpp, SimSYCL)
("Check parallel_for({N1, N2, N3}, some_kernel)",
 "[parallel_for_simplifications]")({
  constexpr int N1 = 2;
  constexpr int N2 = 3;
  constexpr int N3 = 5;
  constexpr int N = N1 * N2 * N3;
  auto action = [=](auto& cgh, auto& acc) {
    cgh.parallel_for({N1, N2, N3}, kernel<3>(acc));
  };
  check<N>(action);
});

}  // namespace parallel_for_simplifications

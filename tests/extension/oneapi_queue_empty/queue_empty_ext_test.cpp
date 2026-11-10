/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/
#include "../../common/common.h"

namespace queue_empty::tests {
constexpr size_t buf_size = 10;
// The kernel below is a single_task loop, so it runs as a single
// work-item. Unoptimized device code (O0/debug builds) runs this loop orders
// of magnitude slower, so the kernel runs for many seconds. A GPU that can be
// preempted only at work-group boundaries then cannot answer an OS preemption
// request in time, and Windows resets the device (TDR). Keep the loop short
// in unoptimized builds; the checks do not depend on the loop length.
#ifdef __OPTIMIZE__
constexpr size_t iter_num_divisor = 1;
#else
constexpr size_t iter_num_divisor = 1000;
#endif
constexpr size_t iter_num = 10000000 / iter_num_divisor;

template <typename AccT>
class Kernel {
  AccT array_acc;
  size_t loop_iter_num;

 public:
  Kernel(AccT acc, size_t n) : array_acc(acc), loop_iter_num(n) {}
  void operator()() const {
    for (size_t i = 0; i < loop_iter_num; i++) {
      int val = sycl::sqrt(float(i));
      array_acc[val % array_acc.size()] = i;
    }
  }
};

TEST_CASE(
    "Test for \"Queue empty\" extension, check extension functionality"
    "when no commands are submitted to the queue",
    "[oneapi_queue_empty]") {
#ifndef SYCL_EXT_ONEAPI_QUEUE_EMPTY
  SKIP("SYCL_EXT_ONEAPI_QUEUE_EMPTY is not defined");
#else
  auto queue = sycl_cts::util::get_cts_object::queue();
  auto is_empty = queue.ext_oneapi_empty();
  CHECK(std::is_same_v<decltype(is_empty), bool>);
  CHECK(is_empty);
#endif
}

TEST_CASE(
    "Test for \"Queue empty\" extension, check extension functionality"
    "after commands are submitted to the queue, test 1",
    "[oneapi_queue_empty]") {
#ifndef SYCL_EXT_ONEAPI_QUEUE_EMPTY
  SKIP("SYCL_EXT_ONEAPI_QUEUE_EMPTY is not defined");
#else
  auto queue = sycl_cts::util::get_cts_object::queue();

  sycl::buffer<size_t, 1> array_buf(buf_size);

  auto e = queue.submit([&](sycl::handler& cgh) {
    auto array_acc = sycl::accessor(array_buf, cgh, sycl::write_only);
    cgh.single_task(Kernel<decltype(array_acc)>(array_acc, iter_num));
  });

  bool is_empty = queue.ext_oneapi_empty();
  if (e.get_info<sycl::info::event::command_execution_status>() !=
      sycl::info::event_command_status::complete) {
    CHECK(!is_empty);
  }

  queue.wait();

  is_empty = queue.ext_oneapi_empty();
  CHECK(is_empty);
#endif
}

TEST_CASE(
    "Test for \"Queue empty\" extension, check extension functionality"
    "after commands are submitted to the queue, test 2",
    "[oneapi_queue_empty]") {
#ifndef SYCL_EXT_ONEAPI_QUEUE_EMPTY
  SKIP("SYCL_EXT_ONEAPI_QUEUE_EMPTY is not defined");
#else
  auto queue = sycl_cts::util::get_cts_object::queue();

  size_t array[buf_size];
  sycl::buffer<size_t, 1> array_buf(array, buf_size);

  auto Ea = queue.submit([&](sycl::handler& cgh) {
    auto array_acc = sycl::accessor(array_buf, cgh, sycl::write_only);
    cgh.single_task(Kernel<decltype(array_acc)>(array_acc, iter_num));
  });

  auto Eb = queue.submit([&](sycl::handler& cgh) {
    cgh.depends_on(Ea);
    auto array_acc = sycl::accessor(array_buf, cgh, sycl::write_only);
    cgh.single_task(Kernel<decltype(array_acc)>(array_acc, iter_num));
  });

  Ea.wait();

  bool is_empty = queue.ext_oneapi_empty();
  if (Eb.get_info<sycl::info::event::command_execution_status>() !=
      sycl::info::event_command_status::complete) {
    CHECK(!is_empty);
  }

  Eb.wait();

  is_empty = queue.ext_oneapi_empty();
  CHECK(is_empty);
#endif
}
}  // namespace queue_empty::tests

/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2023 The Khronos Group Inc.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
*******************************************************************************/
#include "../../common/common.h"

namespace queue_empty::tests {
constexpr size_t buf_size = 10;
constexpr size_t iter_num = 10000000;

template <size_t iter_num, typename AccT>
class Kernel {
  AccT array_acc;

 public:
  Kernel(AccT acc) : array_acc(acc) {}
  void operator()() const {
    for (size_t i = 0; i < iter_num; i++) {
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

  size_t array[buf_size];
  sycl::buffer<size_t, 1> array_buf(array, buf_size);

  auto e = queue.submit([&](sycl::handler& cgh) {
    auto array_acc =
        array_buf.template get_access<sycl::access_mode::write>(cgh);
    cgh.single_task(Kernel<iter_num, decltype(array_acc)>(array_acc));
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
    auto array_acc =
        array_buf.template get_access<sycl::access_mode::write>(cgh);
    cgh.single_task(Kernel<iter_num, decltype(array_acc)>(array_acc));
  });

  auto Eb = queue.submit([&](sycl::handler& cgh) {
    cgh.depends_on(Ea);
    auto array_acc =
        array_buf.template get_access<sycl::access_mode::write>(cgh);
    cgh.single_task(Kernel<iter_num, decltype(array_acc)>(array_acc));
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

/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2022 The Khronos Group Inc.
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

#ifndef SYCL_CTS_QUEUE_QUEUE_SHORTCUTS_EXPLICIT_H
#define SYCL_CTS_QUEUE_QUEUE_SHORTCUTS_EXPLICIT_H

#include "../common/common.h"
#include "../common/disabled_for_test_case.h"
#include "../common/get_cts_object.h"
#include "../common/value_operations.h"
#include "queue_shortcuts_common.h"

namespace queue_shortcuts_explict {

using namespace queue_shortcuts_common;

template <typename T>
class simple_kernel0;
template <typename T>
class simple_kernel1;
template <typename T>
class simple_kernel2;

template <typename T>
void test_explicit_copy(sycl::queue q, unsigned int element_count) {
  const T t_test{1};  // to initialize the sequence: t_test, ++t_test, etc.
  const sycl::range<1> range = sycl::range<1>(element_count);

  // copy (host raw pointer, accessor)
  {
    std::unique_ptr<T[]> src = std::make_unique<T[]>(element_count);
    iota_comp(src.get(), src.get() + element_count, t_test);

    sycl::buffer<T, 1> buffer(range);
    sycl::accessor accessor(buffer, range, sycl::write_only);
    sycl::event e = q.copy(src.get(), accessor);
    e.wait();

    sycl::host_accessor host_accessor(buffer, sycl::read_only);
    std::vector<T> expected(element_count);
    iota_comp(expected.begin(), expected.end(), t_test);
    CHECK(value_operations::are_equal(host_accessor, expected));
  }
  // copy (host shared pointer, accessor)
  {
    std::shared_ptr<T> src(new T[element_count]);
    iota_comp(src.get(), src.get() + element_count, t_test);

    sycl::buffer<T, 1> buffer(range);
    sycl::accessor accessor(buffer, range, sycl::write_only);
    sycl::event e = q.copy(src, accessor);
    e.wait();

    sycl::host_accessor host_accessor(buffer, sycl::read_only);
    std::vector<T> expected(element_count);
    iota_comp(expected.begin(), expected.end(), t_test);
    CHECK(value_operations::are_equal(host_accessor, expected));
  }
  // ComputeCpp requested kernel name could not be found
#ifndef SYCL_CTS_COMPILING_WITH_COMPUTECPP
  //  copy (accessor, host raw pointer)
  {
    sycl::buffer<T, 1> buffer(range);
    q.submit([&](sycl::handler& cgh) {
      sycl::accessor accessor(buffer, cgh, sycl::write_only);
      cgh.parallel_for<simple_kernel0<T>>(element_count, [=](sycl::id<1> idx) {
        accessor[idx] = t_test + static_cast<int>(idx[0]);
      });
    });
    q.wait();
    sycl::accessor accessor(buffer);

    std::unique_ptr<T[]> dest = std::make_unique<T[]>(element_count);
    sycl::event e = q.copy(accessor, dest.get());
    e.wait();

    sycl::buffer<T, 1> dest_buffer(dest.get(), range);
    sycl::host_accessor host_accessor(dest_buffer, sycl::read_only);
    std::vector<T> expected(element_count);
    iota_comp(expected.begin(), expected.end(), t_test);
    CHECK(value_operations::are_equal(host_accessor, expected));
  }
  // copy (accessor, host shared pointer)
  {
    sycl::buffer<T, 1> buffer(range);
    q.submit([&](sycl::handler& cgh) {
      sycl::accessor accessor(buffer, cgh, sycl::write_only);
      cgh.parallel_for<simple_kernel1<T>>(element_count, [=](sycl::id<1> idx) {
        accessor[idx] = t_test + static_cast<int>(idx[0]);
      });
    });
    q.wait();
    sycl::accessor accessor(buffer);

    std::shared_ptr<T> dest(new T[element_count]);
    sycl::event e = q.copy(accessor, dest);
    e.wait();

    sycl::buffer<T, 1> dest_buffer(dest, range);
    sycl::host_accessor host_accessor(dest_buffer, sycl::read_only);
    std::vector<T> expected(element_count);
    iota_comp(expected.begin(), expected.end(), t_test);
    CHECK(value_operations::are_equal(host_accessor, expected));
  }
  // copy (accessor, accessor)
  {
    sycl::buffer<T, 1> buffer_src(range);
    q.submit([&](sycl::handler& cgh) {
      sycl::accessor accessor{buffer_src, cgh, sycl::write_only};
      cgh.parallel_for<simple_kernel2<T>>(element_count, [=](sycl::id<1> idx) {
        accessor[idx] = t_test + static_cast<int>(idx[0]);
      });
    });
    q.wait();

    sycl::buffer<T> buffer_dest(range);
    sycl::event e =
        q.copy(sycl::accessor(buffer_src), sycl::accessor(buffer_dest));
    e.wait();

    sycl::host_accessor host_accessor(buffer_dest, sycl::read_only);
    std::vector<T> expected(element_count);
    iota_comp(expected.begin(), expected.end(), t_test);
    CHECK(value_operations::are_equal(host_accessor, expected));
  }
#else
  WARN(
      "queue.copy() test does not compile for ComputeCPP"
      "Skipping the test case.");
#endif  // SYCL_CTS_COMPILING_WITH_COMPUTECPP
  // ComputeCpp gives an error next time the function is called
#ifndef SYCL_CTS_COMPILING_WITH_COMPUTECPP
  //  update_host
  {
    std::unique_ptr<T[]> src = std::make_unique<T[]>(element_count);
    sycl::buffer<T, 1> buffer(src.get(), range);
    sycl::accessor accessor(buffer, range, sycl::write_only);
    sycl::event e = q.update_host(accessor);
    e.wait();
  }
#else
  WARN(
      "queue.update_host() test does not compile for ComputeCPP"
      "Skipping the test case.");
#endif
  // ComputeCpp function not defined
#ifndef SYCL_CTS_COMPILING_WITH_COMPUTECPP
  //  fill
  {
    sycl::buffer<T, 1> buffer(range);
    sycl::accessor accessor(buffer, range, sycl::write_only);
    sycl::event e = q.fill(accessor, t_test);
    e.wait();

    sycl::host_accessor host_accessor(buffer, sycl::read_only);
    CHECK(value_operations::are_equal(host_accessor, t_test));
  }
#else
  WARN(
      "queue.fill() test does not compile for ComputeCPP"
      "Skipping the test case.");
#endif  // SYCL_CTS_COMPILING_WITH_COMPUTECPP
}

template <typename T>
class check_queue_shortcuts_explicit_for_type {
  static constexpr unsigned int element_count = 10;

 public:
  void operator()(sycl::queue queue, const std::string& type_name) {
    INFO("for type \"" << type_name << "\": ");

    test_explicit_copy<T>(queue, element_count);
  }
};

}  // namespace queue_shortcuts_explict

#endif  // SYCL_CTS_QUEUE_QUEUE_SHORTCUTS_EXPLICIT_H

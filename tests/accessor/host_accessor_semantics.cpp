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

#include "../common/common.h"
#include "../common/disabled_for_test_case.h"
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
#include "../common/semantics_reference.h"
#endif

template <int Dimensions>
struct storage {
  std::size_t byte_size;
  std::size_t size;
  std::size_t max_size;
  bool is_empty;
  sycl::range<Dimensions> range =
      sycl_cts::util::get_cts_object::range<Dimensions>::get(1, 1, 1);
  sycl::id<Dimensions> offset;

  template <typename T, sycl::access_mode AccessMode>
  explicit storage(
      const sycl::host_accessor<T, Dimensions, AccessMode>& host_accessor)
      : byte_size(host_accessor.byte_size()),
        size(host_accessor.size()),
        max_size(host_accessor.max_size()),
        is_empty(host_accessor.empty()),
        range(host_accessor.get_range()),
        offset(host_accessor.get_offset()) {}

  template <typename T, sycl::access_mode AccessMode>
  bool check(const sycl::host_accessor<T, Dimensions, AccessMode>&
                 host_accessor) const {
    return host_accessor.byte_size() == byte_size &&
           host_accessor.size() == size &&
           host_accessor.max_size() == max_size &&
           host_accessor.empty() == is_empty &&
           host_accessor.get_range() == range &&
           host_accessor.get_offset() == offset;
  }
};

DISABLED_FOR_TEST_CASE(hipSYCL)
("host_accessor common reference semantics", "[host_accessor]")({
#if defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
  WARN(
      "Implementation does not define std::hash for accessor. "
      "Affected test cases set to fail.");
#endif

  sycl::buffer<int> buffer_0{sycl::range<1>{1}};
  sycl::host_accessor<int> host_accessor_0{buffer_0};

  sycl::buffer<int> buffer_1{sycl::range<1>{1}};
  sycl::host_accessor<int> host_accessor_1{buffer_1};
  common_reference_semantics::check_host<storage<1>>(
      host_accessor_0, host_accessor_1, "host_accessor");
});

DISABLED_FOR_TEST_CASE(hipSYCL)
("host_accessor common reference semantics, mutation", "[host_accessor]")({
  constexpr int val = 1;
  constexpr int new_val = 2;
  sycl::buffer<int> buffer{sycl::range<1>{1}};
  sycl::host_accessor<int> t0{buffer};

  SECTION("mutation to copy") {
    t0[0] = val;
    sycl::host_accessor<int> t1(t0);
    t1[0] = new_val;
    CHECK(new_val == t0[0]);
  }

  SECTION("mutation to original") {
    t0[0] = val;
    sycl::host_accessor<int> t1(t0);
    t0[0] = new_val;
    CHECK(new_val == t1[0]);
  }

  SECTION("mutation to original, const copy") {
    t0[0] = val;
    sycl::host_accessor<int> t1(t0);
    t0[0] = new_val;
    CHECK(new_val == t1[0]);
  }
});

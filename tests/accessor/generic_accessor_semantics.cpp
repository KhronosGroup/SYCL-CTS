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
#include "../common/semantics_reference.h"

template <int Dimensions>
struct storage {
  bool is_placeholder;
  std::size_t byte_size;
  std::size_t size;
  std::size_t max_size;
  bool is_empty;
  sycl::range<Dimensions> range;
  sycl::id<Dimensions> offset;

  template <typename T, sycl::access_mode AccessMode, sycl::target AccessTarget,
            sycl::access::placeholder IsPlaceHolder>
  explicit storage(const sycl::accessor<T, Dimensions, AccessMode, AccessTarget,
                               IsPlaceHolder>& accessor)
      : is_placeholder(accessor.is_placeholder()),
        byte_size(accessor.byte_size()),
        size(accessor.size()),
        max_size(accessor.max_size()),
        is_empty(accessor.empty()),
        range(accessor.get_range()),
        offset(accessor.get_offset()) {}

  template <typename T, sycl::access_mode AccessMode, sycl::target AccessTarget,
            sycl::access::placeholder IsPlaceHolder>
  bool check(const sycl::accessor<T, Dimensions, AccessMode, AccessTarget,
                                  IsPlaceHolder>& accessor) const {
    return accessor.is_placeholder() == is_placeholder &&
           accessor.byte_size() == byte_size && accessor.size() == size &&
           accessor.max_size() == max_size && accessor.empty() == is_empty &&
           accessor.get_range() == range && accessor.get_offset() == offset;
  }
};

// DPCPP has no member 'host_task' in 'sycl::access::target'.
DISABLED_FOR_TEST_CASE(DPCPP)
("generic accessor common reference semantics (host)", "[accessor]")({
  {  // target::host_task
    int val_0;
    sycl::buffer<int> buffer_0{&val_0, sycl::range<1>{1}};
    sycl::accessor<int, 1, sycl::access_mode::read_write,
                   sycl::target::host_task>
        accessor_0{buffer_0};

    int val_1;
    sycl::buffer<int> buffer_1{&val_1, sycl::range<1>{1}};
    sycl::accessor<int, 1, sycl::access_mode::read_write,
                   sycl::target::host_task>
        accessor_1{buffer_1};

    common_reference_semantics::check_host<storage<1>>(
        accessor_0, accessor_1,
        "accessor<int, 1, access_mode::read_write, target::host_task>");
}
{  // target::device
  sycl::buffer<int> buffer_0{sycl::range<1>{1}};
  sycl::buffer<int> buffer_1{sycl::range<1>{1}};
  sycl_cts::util::get_cts_object::queue().submit([&](sycl::handler& cgh) {
    auto accessor_0 = buffer_0.get_access<sycl::access_mode::read_write,
                                          sycl::target::device>(cgh);
    auto accessor_1 = buffer_1.get_access<sycl::access_mode::read_write,
                                          sycl::target::device>(cgh);
    common_reference_semantics::check_host<storage<1>>(
        accessor_0, accessor_1,
        "accessor<int, 1, access_mode::read_write, target::device>");
  });
}
});

// DPCPP has no member 'host_task' in 'sycl::access::target'.
DISABLED_FOR_TEST_CASE(DPCPP)
("generic accessor common reference semantics, mutation (host)", "[accessor]")({
  constexpr int new_val = 2;
  int val = 1;
  sycl::buffer<int> buffer{&val, sycl::range<1>{1}};
  sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::target::host_task>
      t0{buffer};

  SECTION("mutation to copy") {
    sycl::accessor<int, 1, sycl::access_mode::read_write,
                   sycl::target::host_task>
        t1(t0);
    t1[0] = new_val;
    CHECK(new_val == t0[0]);
  }

  SECTION("mutation to original") {
    sycl::accessor<int, 1, sycl::access_mode::read_write,
                   sycl::target::host_task>
        t1(t0);
    t0[0] = new_val;
    CHECK(new_val == t1[0]);
  }

  SECTION("mutation to original, const copy") {
    const sycl::accessor<int, 1, sycl::access_mode::read_write,
                         sycl::target::host_task>
        t1(t0);
    t0[0] = new_val;
    CHECK(new_val == t1[0]);
  }
});

TEST_CASE("generic accessor common reference semantics (kernel)",
          "[accessor]") {
  sycl::buffer<int> buffer{sycl::range<1>{1}};
  using type =
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::target::device, sycl::access::placeholder::false_t>;
  common_reference_semantics::check_kernel<storage<1>, type>(
      [&](sycl::handler& cgh) {
        return buffer
            .get_access<sycl::access_mode::read_write, sycl::target::device>(
                cgh);
      },
      "accessor<int, 1, access_mode::read_write, target::device>");
}

template <int TestCase>
class kernel_name_generic;

TEST_CASE("generic accessor common reference semantics, mutation (kernel)",
          "[accessor]") {
  sycl::queue queue = sycl_cts::util::get_cts_object::queue();
  int result = 0;

#ifdef SYCL_CTS_COMPILING_WITH_COMPUTECPP
  int val = 1;
  int new_val = 2;
#else
  constexpr int val = 1;
  constexpr int new_val = 2;
#endif
  sycl::buffer<int> buffer{sycl::range<1>{1}};

  SECTION("mutation to copy") {
    {
      sycl::buffer<int> buffer_result{&result, sycl::range<1>{1}};
      queue.submit([&](sycl::handler& cgh) {
        auto acc_result = buffer_result.get_access(cgh);
        sycl::accessor<int, 1> t0 = buffer.get_access(cgh);
        cgh.single_task<kernel_name_generic<0>>([=]() {
          t0[0] = val;
          sycl::accessor<int, 1> t1(t0);
          t1[0] = new_val;
          acc_result[0] = t0[0];
        });
      });
    }
    CHECK(new_val == result);
  }

  SECTION("mutation to original") {
    {
      sycl::buffer<int> buffer_result{&result, sycl::range<1>{1}};
      queue.submit([&](sycl::handler& cgh) {
        auto acc_result = buffer_result.get_access(cgh);
        sycl::accessor<int, 1> t0 = buffer.get_access(cgh);
        cgh.single_task<kernel_name_generic<1>>([=]() {
          t0[0] = val;
          sycl::accessor<int, 1> t1(t0);
          t0[0] = new_val;
          acc_result[0] = t1[0];
        });
      });
    }
    CHECK(new_val == result);
  }

  SECTION("mutation to original, const copy") {
    {
      sycl::buffer<int> buffer_result{&result, sycl::range<1>{1}};
      queue.submit([&](sycl::handler& cgh) {
        auto acc_result = buffer_result.get_access(cgh);
        sycl::accessor<int, 1> t0 = buffer.get_access(cgh);
        cgh.single_task<kernel_name_generic<2>>([=]() {
          t0[0] = val;
          const sycl::accessor<int, 1> t1(t0);
          t0[0] = new_val;
          acc_result[0] = t1[0];
        });
      });
    }
    CHECK(new_val == result);
  }
}

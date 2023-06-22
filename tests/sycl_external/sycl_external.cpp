/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2022-2023 The Khronos Group Inc.
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

constexpr int value = 42;

using accT = sycl::accessor<int, 1>;
using hostAccT = sycl::host_accessor<int, 1>;

#ifdef SYCL_EXTERNAL
SYCL_EXTERNAL void same_unit_device(const accT& acc) { acc[0] = value; }
SYCL_EXTERNAL void same_unit_host(hostAccT& acc) { acc[0] = value; }
template <int TestCase>
class kernel_same_unit {
  accT acc;

 public:
  kernel_same_unit(accT input_acc) : acc(input_acc) {}
  void operator()() const { same_unit_device(acc); }
  void operator()(sycl::item<1> item) const { same_unit_device(acc); }
};

SYCL_EXTERNAL void simple_separate_unit_device(const accT& acc);
SYCL_EXTERNAL void simple_separate_unit_host(hostAccT& acc);
template <int TestCase>
class kernel_simple_separate_unit {
  accT acc;

 public:
  kernel_simple_separate_unit(accT input_acc) : acc(input_acc) {}
  void operator()() const { simple_separate_unit_device(acc); }
  void operator()(sycl::item<1> item) const {
    simple_separate_unit_device(acc);
  }
};

SYCL_EXTERNAL extern void extern_separate_unit_device(const accT& acc);
SYCL_EXTERNAL extern void extern_separate_unit_host(hostAccT& acc);
template <int TestCase>
class kernel_extern_separate_unit {
  accT acc;

 public:
  kernel_extern_separate_unit(accT input_acc) : acc(input_acc) {}
  void operator()() const { extern_separate_unit_device(acc); }
  void operator()(sycl::item<1> item) const {
    extern_separate_unit_device(acc);
  }
};

template <sycl::aspect aspect>
SYCL_EXTERNAL [[sycl::device_has(aspect)]] void before_aspect_device(
    const accT& acc);
template <sycl::aspect aspect>
SYCL_EXTERNAL [[sycl::device_has(aspect)]] void before_aspect_host(
    hostAccT& acc);
template <int TestCase, sycl::aspect aspect>
class kernel_before_aspect {
  accT acc;

 public:
  kernel_before_aspect(accT input_acc) : acc(input_acc) {}
  void operator()() const { before_aspect_device<aspect>(acc); }
  void operator()(sycl::item<1> item) const {
    before_aspect_device<aspect>(acc);
  }
};

template <sycl::aspect aspect>
[[sycl::device_has(aspect)]] SYCL_EXTERNAL [[noreturn]] void
between_aspects_device(const accT& acc);
template <sycl::aspect aspect>
[[sycl::device_has(aspect)]] SYCL_EXTERNAL [[noreturn]] void
between_aspects_host(hostAccT& acc);
template <int TestCase, sycl::aspect aspect>
class kernel_between_aspects {
  accT acc;

 public:
  kernel_between_aspects(accT input_acc) : acc(input_acc) {}
  void operator()() const { between_aspects_device<aspect>(acc); }
  void operator()(sycl::item<1> item) const {
    between_aspects_device<aspect>(acc);
  }
};

template <sycl::aspect aspect>
[[sycl::device_has(aspect)]] SYCL_EXTERNAL void after_aspect_device(
    const accT& acc);
template <sycl::aspect aspect>
[[sycl::device_has(aspect)]] SYCL_EXTERNAL void after_aspect_host(
    hostAccT& acc);
template <int TestCase, sycl::aspect aspect>
class kernel_after_aspect {
  accT acc;

 public:
  kernel_after_aspect(accT input_acc) : acc(input_acc) {}
  void operator()() const { after_aspect_device<aspect>(acc); }
  void operator()(sycl::item<1> item) const {
    after_aspect_device<aspect>(acc);
  }
};
#endif  // SYCL_EXTERNAL

void test_host(void (*f)(hostAccT&)) {
  SECTION("Check call from host code") {
    sycl::buffer<int> buf{1};
    hostAccT acc{buf};
    f(acc);
    CHECK(acc[0] == value);
  }
}

template <template <int, sycl::aspect...> typename kernel,
          sycl::aspect... aspects>
void test_device() {
  auto queue = sycl_cts::util::get_cts_object::queue();
  int data = 0;
  SECTION("Check call from single_task") {
    {
      sycl::buffer<int> buf{&data, {1}};
      queue
          .submit([&](sycl::handler& cgh) {
            accT acc{buf, cgh};
            cgh.single_task<kernel<1, aspects...>>(kernel<1, aspects...>(acc));
          })
          .wait_and_throw();
    }
    CHECK(data == value);
  }
  SECTION("Check call from parallel_for") {
    {
      sycl::buffer<int> buf{&data, {1}};
      queue
          .submit([&](sycl::handler& cgh) {
            accT acc{buf, cgh};
            cgh.parallel_for<kernel<2, aspects...>>(1,
                                                    kernel<2, aspects...>(acc));
          })
          .wait_and_throw();
    }
    CHECK(data == value);
  }
}

TEST_CASE("Function with SYCL_EXTERNAL is defined in the same translation unit",
          "[sycl_external]") {
#ifdef SYCL_EXTERNAL
  test_host(same_unit_host);
  test_device<kernel_same_unit>();
#else
  SKIP("SYCL_EXTERNAL is not supported");
#endif
}

TEST_CASE(
    "Function with SYCL_EXTERNAL is defined in different translation unit",
    "[sycl_external]") {
#ifdef SYCL_EXTERNAL
  test_host(simple_separate_unit_host);
  test_device<kernel_simple_separate_unit>();
#else
  SKIP("SYCL_EXTERNAL is not supported");
#endif
}

TEST_CASE("Function with SYCL_EXTERNAL is declared with keyword extern",
          "[sycl_external]") {
#ifdef SYCL_EXTERNAL
  test_host(extern_separate_unit_host);
  test_device<kernel_extern_separate_unit>();
#else
  SKIP("SYCL_EXTERNAL is not supported");
#endif
}

TEST_CASE("Function with SYCL_EXTERNAL that decalred before attribute",
          "[sycl_external]") {
#ifdef SYCL_EXTERNAL
  auto queue = sycl_cts::util::get_cts_object::queue();
  if (queue.get_device().has(sycl::aspect::cpu)) {
    test_host(before_aspect_host<sycl::aspect::cpu>);
    test_device<kernel_before_aspect, sycl::aspect::cpu>();
  } else if (queue.get_device().has(sycl::aspect::gpu)) {
    test_host(before_aspect_host<sycl::aspect::gpu>);
    test_device<kernel_before_aspect, sycl::aspect::gpu>();
  } else if (queue.get_device().has(sycl::aspect::accelerator)) {
    test_host(before_aspect_host<sycl::aspect::accelerator>);
    test_device<kernel_before_aspect, sycl::aspect::accelerator>();
  }
#else
  SKIP("SYCL_EXTERNAL is not supported");
#endif
}

TEST_CASE("Function with SYCL_EXTERNAL that decalred between attributes",
          "[sycl_external]") {
#ifdef SYCL_EXTERNAL
  auto queue = sycl_cts::util::get_cts_object::queue();
  if (queue.get_device().has(sycl::aspect::cpu)) {
    test_host(between_aspects_host<sycl::aspect::cpu>);
    test_device<kernel_between_aspects, sycl::aspect::cpu>();
  } else if (queue.get_device().has(sycl::aspect::gpu)) {
    test_host(between_aspects_host<sycl::aspect::gpu>);
    test_device<kernel_between_aspects, sycl::aspect::gpu>();
  } else if (queue.get_device().has(sycl::aspect::accelerator)) {
    test_host(between_aspects_host<sycl::aspect::accelerator>);
    test_device<kernel_between_aspects, sycl::aspect::accelerator>();
  }
#else
  SKIP("SYCL_EXTERNAL is not supported");
#endif
}

TEST_CASE("Function with SYCL_EXTERNAL that decalred after attribute",
          "[sycl_external]") {
#ifdef SYCL_EXTERNAL
  auto queue = sycl_cts::util::get_cts_object::queue();
  if (queue.get_device().has(sycl::aspect::cpu)) {
    test_host(after_aspect_host<sycl::aspect::cpu>);
    test_device<kernel_after_aspect, sycl::aspect::cpu>();
  } else if (queue.get_device().has(sycl::aspect::gpu)) {
    test_host(after_aspect_host<sycl::aspect::gpu>);
    test_device<kernel_after_aspect, sycl::aspect::gpu>();
  } else if (queue.get_device().has(sycl::aspect::accelerator)) {
    test_host(after_aspect_host<sycl::aspect::accelerator>);
    test_device<kernel_after_aspect, sycl::aspect::accelerator>();
  }
#else
  SKIP("SYCL_EXTERNAL is not supported");
#endif
}

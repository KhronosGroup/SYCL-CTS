/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2024 The Khronos Group Inc.
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

namespace local_memory::tests {

#ifdef SYCL_EXT_ONEAPI_LOCAL_MEMORY

struct Point3D {
  int x;
  int y;
  int z;
  Point3D(int x = 1, int y = -2, int z = 3) : x{x}, y{y}, z{z} {}
};

static bool operator==(const Point3D& a, const Point3D& b) {
  return (a.x == b.x) && (a.y == b.y) && (a.z == b.z);
}

template <typename T, typename... Args>
static bool checkInitialValue(const T& value, Args&&... args) {
  const T reference(std::forward<Args>(args)...);
  return (value == reference);
}

template <typename T, int N, typename... Args>
static bool checkInitialValue(const T (&value)[N], Args&&... args) {
  const T reference[N]{std::forward<Args>(args)...};
  for (int i = 0; i < N; i++)
    if (value[i] != reference[i]) return false;
  return true;
}

template <typename F, size_t LocalSize = 10, size_t GlobalSize = 10 * LocalSize>
static void runTestKernel(const F& kernel) {
  sycl::queue q;
  bool* passed = sycl::malloc_shared<bool>(GlobalSize, q);
  q.parallel_for(sycl::nd_range<1>(GlobalSize, LocalSize),
                 [=](sycl::nd_item<1> item) { kernel(item, passed); })
      .wait();
  for (int i = 0; i < GlobalSize; i++) CHECK(passed[i]);
  sycl::free(passed, q);
}

template <typename T, typename... Args>
static void testInitialValue(Args&&... args) {
  runTestKernel([=](sycl::nd_item<1> item, bool* passed) {
    auto ptr =
        sycl::ext::oneapi::group_local_memory<T>(item.get_group(), args...);
    static_assert(
        std::is_same_v<
            decltype(ptr),
            sycl::multi_ptr<T, sycl::access::address_space::local_space>>,
        "group_local_memory returns the wrong type");
    passed[item.get_global_id()] = checkInitialValue(*ptr, args...);
  });
}

template <typename T>
static void testInitialValueForOverwrite() {
  runTestKernel([=](sycl::nd_item<1> item, bool* passed) {
    auto ptr = sycl::ext::oneapi::group_local_memory_for_overwrite<T>(
        item.get_group());
    static_assert(
        std::is_same_v<
            decltype(ptr),
            sycl::multi_ptr<T, sycl::access::address_space::local_space>>,
        "group_local_memory_for_overwrite returns the wrong type");
    passed[item.get_global_id()] =
        (std::is_same_v<T, Point3D> ? checkInitialValue(*ptr) : true);
  });
}

template <typename T, typename... Args>
static void testDifferentInitialValues(Args&&... args) {
  runTestKernel([=](sycl::nd_item<1> item, bool* passed) {
    const int factor = static_cast<int>(item.get_group_linear_id());
    auto ptr = sycl::ext::oneapi::group_local_memory<T>(
        item.get_group(), (static_cast<decltype(args)>(args * factor))...);
    passed[item.get_global_id()] = checkInitialValue(*ptr, (args * factor)...);
  });
}

template <typename T, size_t N, typename... Args>
static void testArrayInitializationWithFewArguments(Args&&... args) {
  runTestKernel([=](sycl::nd_item<1> item, bool* passed) {
    constexpr size_t M = sizeof...(args);
    static_assert(
        M < N,
        "Array must be initialized with fewer arguments than its length");
    auto ptr =
        sycl::ext::oneapi::group_local_memory<T[N]>(item.get_group(), args...);
    passed[item.get_global_id()] = checkInitialValue(*ptr, args...);
  });
}

template <typename T>
static void testLocalMemoryAvailability() {
  constexpr size_t N = 10;
  const auto kernel = [=](sycl::nd_item<1> item, bool* passed) {
    auto array = *sycl::ext::oneapi::group_local_memory<T[N]>(item.get_group());
    array[(N - 1) - item.get_local_linear_id()] = item.get_local_linear_id();
    sycl::group_barrier(item.get_group());
    passed[item.get_global_id()] =
        (array[item.get_local_linear_id()] ==
         static_cast<T>((N - 1) - item.get_local_linear_id()));
  };
  runTestKernel<decltype(kernel), N>(kernel);
}

template <typename T>
static void testLocalMemoryForOverwriteAvailability() {
  constexpr size_t N = 10;
  const auto kernel = [=](sycl::nd_item<1> item, bool* passed) {
    auto array = *sycl::ext::oneapi::group_local_memory_for_overwrite<T[N]>(
        item.get_group());
    array[(N - 1) - item.get_local_linear_id()] = item.get_local_linear_id();
    sycl::group_barrier(item.get_group());
    passed[item.get_global_id()] =
        (array[item.get_local_linear_id()] ==
         static_cast<T>((N - 1) - item.get_local_linear_id()));
  };
  runTestKernel<decltype(kernel), N>(kernel);
}

#endif

TEST_CASE("Test case for \"Local Memory\" extension", "[oneapi_local_memory") {
#ifndef SYCL_EXT_ONEAPI_LOCAL_MEMORY
  SKIP("SYCL_EXT_ONEAPI_LOCAL_MEMORY is not defined");
#else
  testInitialValue<int>(2);
  testInitialValue<float>(1.5f);
  testInitialValue<Point3D>(5);
  testInitialValue<Point3D>(5, 7);
  testInitialValue<Point3D>(5, 7, 13);
  testInitialValue<int[5]>(1, -2, 3, -4, 5);
#ifdef SYCL_CTS_ENABLE_FULL_CONFORMANCE
  testInitialValue<bool>(true);
  testInitialValue<char>('A');
  testInitialValue<signed char>('A');
  testInitialValue<unsigned char>('A');
  testInitialValue<short>(17);
  testInitialValue<unsigned short>(19);
  testInitialValue<unsigned int>(23);
  testInitialValue<long>(29);
  testInitialValue<unsigned long>(31);
  testInitialValue<long long>(37);
  testInitialValue<unsigned long long>(43);
  testInitialValue<double>(3.14);
#endif

  testInitialValueForOverwrite<int>();
  testInitialValueForOverwrite<float>();
  testInitialValueForOverwrite<Point3D>();
  testInitialValueForOverwrite<int[5]>();
#ifdef SYCL_CTS_ENABLE_FULL_CONFORMANCE
  testInitialValueForOverwrite<bool>();
  testInitialValueForOverwrite<char>();
  testInitialValueForOverwrite<signed char>();
  testInitialValueForOverwrite<unsigned char>();
  testInitialValueForOverwrite<short>();
  testInitialValueForOverwrite<unsigned short>();
  testInitialValueForOverwrite<unsigned int>();
  testInitialValueForOverwrite<long>();
  testInitialValueForOverwrite<unsigned long>();
  testInitialValueForOverwrite<long long>();
  testInitialValueForOverwrite<unsigned long long>();
  testInitialValueForOverwrite<double>();
#endif

  testDifferentInitialValues<int>(2);
  testDifferentInitialValues<float>(1.5f);
  testDifferentInitialValues<Point3D>(5);
  testDifferentInitialValues<Point3D>(5, 7);
  testDifferentInitialValues<Point3D>(5, 7, 13);
  testDifferentInitialValues<int[5]>(1, -2, 3, -4, 5);
#ifdef SYCL_CTS_ENABLE_FULL_CONFORMANCE
  testDifferentInitialValues<bool>(true);
  testDifferentInitialValues<char>('A');
  testDifferentInitialValues<signed char>('A');
  testDifferentInitialValues<unsigned char>('A');
  testDifferentInitialValues<short>(17);
  testDifferentInitialValues<unsigned short>(19);
  testDifferentInitialValues<unsigned int>(23);
  testDifferentInitialValues<long>(29);
  testDifferentInitialValues<unsigned long>(31);
  testDifferentInitialValues<long long>(37);
  testDifferentInitialValues<unsigned long long>(43);
  testDifferentInitialValues<double>(3.14);
#endif

  testLocalMemoryAvailability<int>();
  testLocalMemoryAvailability<float>();
  testLocalMemoryAvailability<Point3D>();
#ifdef SYCL_CTS_ENABLE_FULL_CONFORMANCE
  testLocalMemoryAvailability<bool>();
  testLocalMemoryAvailability<char>();
  testLocalMemoryAvailability<signed char>();
  testLocalMemoryAvailability<unsigned char>();
  testLocalMemoryAvailability<short>();
  testLocalMemoryAvailability<unsigned short>();
  testLocalMemoryAvailability<unsigned int>();
  testLocalMemoryAvailability<long>();
  testLocalMemoryAvailability<unsigned long>();
  testLocalMemoryAvailability<long long>();
  testLocalMemoryAvailability<unsigned long long>();
  testLocalMemoryAvailability<double>();
#endif

  testLocalMemoryForOverwriteAvailability<int>();
  testLocalMemoryForOverwriteAvailability<float>();
  testLocalMemoryForOverwriteAvailability<Point3D>();
#ifdef SYCL_CTS_ENABLE_FULL_CONFORMANCE
  testLocalMemoryForOverwriteAvailability<bool>();
  testLocalMemoryForOverwriteAvailability<char>();
  testLocalMemoryForOverwriteAvailability<signed char>();
  testLocalMemoryForOverwriteAvailability<unsigned char>();
  testLocalMemoryForOverwriteAvailability<short>();
  testLocalMemoryForOverwriteAvailability<unsigned short>();
  testLocalMemoryForOverwriteAvailability<unsigned int>();
  testLocalMemoryForOverwriteAvailability<long>();
  testLocalMemoryForOverwriteAvailability<unsigned long>();
  testLocalMemoryForOverwriteAvailability<long long>();
  testLocalMemoryForOverwriteAvailability<unsigned long long>();
  testLocalMemoryForOverwriteAvailability<double>();
#endif

  testArrayInitializationWithFewArguments<int, 5>(1, -2, 3, -4);
#endif
}

}  // namespace local_memory::tests

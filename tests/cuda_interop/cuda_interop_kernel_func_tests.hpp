/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
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

#ifndef CUDA_INTEROP_KERNEL_FUNC_TESTS
#define CUDA_INTEROP_KERNEL_FUNC_TESTS

#include "../common/common.h"
#include "../common/type_coverage.h"

// User defined struct
struct s1 {
  float A;
  double B;
  int C;
  long D;
  bool E;
};

// Class with user-defined default constructor
class c1 {
 public:
  float A;
  double B;
  int C;
  long D;
  bool E;

  c1() {
    A = 1;
    B = 2;
    C = 3;
    D = 4;
    E = false;
  };
};

// Class with deleted default constructor and user-defined constructor
class c2 {
 public:
  float A;
  double B;
  int C;
  long D;
  bool E;

  c2() = delete;
  c2(float a, double b, int c, long d, bool e) : A(a), B(b), C(c), D(d), E(e){};
};

static auto get_types() {
  static const auto types = named_type_pack<
      char, signed char, unsigned char, short int, unsigned short int, int,
      unsigned int, long int, unsigned long int, long long int,
      unsigned long long int, float, bool, std::byte, std::int8_t, std::int16_t,
      std::int32_t, std::int64_t, std::uint8_t, std::uint16_t, std::uint32_t,
      std::uint64_t, std::size_t>::generate("char", "signed char",
                                            "unsigned char", "short int",
                                            "unsigned short int", "int",
                                            "unsigned int", "long int",
                                            "unsigned long int",
                                            "long long int",
                                            "unsigned long long int", "float",
                                            "bool", "std::byte", "std::int8_t",
                                            "std::int16_t", "std::int32_t",
                                            "std::int64_t", "std::uint8_t",
                                            "std::uint16_t", "std::uint32_t",
                                            "std::uint64_t", "std::size_t");
  return types;
}

template <typename T>
class kernel_accessor;

/** check get_native() returns the correct type for an accessor
 */
template <typename T>
struct test_accessor {
  void operator()(sycl::queue &queue, const std::string &typeName) {
    size_t constexpr size = 1;
    T data[size];
    bool is_type_correct[size] = {false};
    {
      sycl::buffer<T> buff(data, sycl::range<1>(size));
      sycl::buffer<bool> is_type_buff(is_type_correct, sycl::range<1>(size));

      queue.submit([&](sycl::handler &cgh) {
        auto acc = buff.template get_access<sycl::access::mode::read>(cgh);
        auto is_type_acc =
            is_type_buff.get_access<sycl::access::mode::write>(cgh);

        cgh.single_task<kernel_accessor<T>>([=] {
          auto native_handle = sycl::get_native<sycl::backend::cuda>(acc);
          is_type_acc[0] = std::is_same_v<decltype(native_handle), T *>;
        });
      });
    }

    INFO("Check CUDA kernel function accessor interop with \"" + typeName +
         "\" type");
    CHECK(is_type_correct[0]);
  }
};

template <typename T>
class kernel_constant_buffer_accessor;

/** check get_native() returns the correct type for a constant-buffer accessor
 */
template <typename T>
struct test_constant_buffer_accessor {
  void operator()(sycl::queue &queue, const std::string &typeName) {
    size_t constexpr size = 1;
    T data[size];
    bool is_type_correct[size] = {false};
    {
      sycl::buffer<T> buff(data, sycl::range<1>(size));
      sycl::buffer<bool> is_type_buff(is_type_correct, sycl::range<1>(size));

      queue.submit([&](sycl::handler &cgh) {
        auto acc = buff.template get_access<sycl::access::mode::read,
                                            sycl::target::constant_buffer>(cgh);
        auto is_type_acc =
            is_type_buff.get_access<sycl::access::mode::write>(cgh);

        cgh.single_task<kernel_constant_buffer_accessor<T>>([=] {
          auto native_handle = sycl::get_native<sycl::backend::cuda>(acc);
          is_type_acc[0] = std::is_same_v<decltype(native_handle), T *>;
        });
      });
    }

    INFO("Check CUDA kernel function constant buffer accessor interop with \"" +
         typeName + "\" type");
    CHECK(is_type_correct[0]);
  }
};

template <typename T>
class kernel_local_target_accessor;

/** check get_native() returns the correct type for an accessor with
 * target::local
 */
template <typename T>
void test_local_target_accessor(sycl::queue &queue,
                                const std::string &typeName) {
  size_t constexpr size = 1;
  bool is_type_correct[size] = {false};
  {
    sycl::buffer<bool> is_type_buff(is_type_correct, sycl::range<1>(size));

    queue.submit([&](sycl::handler &cgh) {
      auto acc =
          sycl::accessor<T, 1, sycl::access::mode::write, sycl::target::local>(
              sycl::range<1>(size), cgh);
      auto is_type_acc =
          is_type_buff.get_access<sycl::access::mode::write>(cgh);

      cgh.parallel_for_work_group<kernel_local_target_accessor<T>>(
          sycl::range<1>(1), sycl::range<1>(1), [=](sycl::group<1>) {
            auto native_handle = sycl::get_native<sycl::backend::cuda>(acc);
            is_type_acc[0] = std::is_same_v<decltype(native_handle), T *>;
          });
    });
  }

  INFO(
      "Check CUDA kernel function accessor with target::local interop with \"" +
      typeName + "\" type");
  CHECK(is_type_correct[0]);
}

template <typename T>
class kernel_local_accessor;

/** check get_native() returns the correct type for a local_accessor
 */
template <typename T>
void test_local_accessor(sycl::queue &queue, const std::string &typeName) {
  size_t constexpr size = 1;
  bool is_type_correct[size] = {false};
  {
    sycl::buffer<bool> is_type_buff(is_type_correct, sycl::range<1>(size));

    queue.submit([&](sycl::handler &cgh) {
      auto acc = sycl::local_accessor<T, 1>(sycl::range<1>(size), cgh);
      auto is_type_acc =
          is_type_buff.get_access<sycl::access::mode::write>(cgh);

      cgh.parallel_for_work_group<kernel_local_accessor<T>>(
          sycl::range<1>(1), sycl::range<1>(1), [=](sycl::group<1>) {
            auto native_handle = sycl::get_native<sycl::backend::cuda>(acc);
            is_type_acc[0] = std::is_same_v<decltype(native_handle), T *>;
          });
    });
  }

  INFO("Check CUDA kernel function local_accessor interop with \"" + typeName +
       "\" type");
  CHECK(is_type_correct[0]);
}

/** check get_native() returns the correct type for a local_accessor and
 * target::local
 */
template <typename T>
struct test_local {
  void operator()(sycl::queue &queue, const std::string &typeName) {
    test_local_target_accessor<T>(queue, typeName);
    test_local_accessor<T>(queue, typeName);
  }
};

/** check get_native() returns the correct type for an accessor
 */
template <>
struct test_accessor<c2> {
  void operator()(sycl::queue &queue, const std::string &typeName) {
    size_t constexpr size = 1;
    c2 data[size] = {c2(1, 2, 3, 4, false)};
    bool is_type_correct[size] = {false};
    {
      sycl::buffer<c2> buff(data, sycl::range<1>(size));
      sycl::buffer<bool> is_type_buff(is_type_correct, sycl::range<1>(size));

      queue.submit([&](sycl::handler &cgh) {
        auto acc = buff.template get_access<sycl::access::mode::read>(cgh);
        auto is_type_acc =
            is_type_buff.get_access<sycl::access::mode::write>(cgh);

        cgh.single_task<kernel_accessor<c2>>([=] {
          auto native_handle = sycl::get_native<sycl::backend::cuda>(acc);
          is_type_acc[0] = std::is_same_v<decltype(native_handle), c2 *>;
        });
      });
    }

    INFO("Check CUDA kernel function accessor interop with \"" + typeName +
         "\" type");
    CHECK(is_type_correct[0]);
  }
};

/** check get_native() returns the correct type for a constant-buffer accessor
 */
template <>
struct test_constant_buffer_accessor<c2> {
  void operator()(sycl::queue &queue, const std::string &typeName) {
    size_t constexpr size = 1;
    c2 data[size] = {c2(1, 2, 3, 4, false)};
    bool is_type_correct[size] = {false};
    {
      sycl::buffer<c2> buff(data, sycl::range<1>(size));
      sycl::buffer<bool> is_type_buff(is_type_correct, sycl::range<1>(size));

      queue.submit([&](sycl::handler &cgh) {
        auto acc = buff.template get_access<sycl::access::mode::read,
                                            sycl::target::constant_buffer>(cgh);
        auto is_type_acc =
            is_type_buff.get_access<sycl::access::mode::write>(cgh);

        cgh.single_task<kernel_constant_buffer_accessor<c2>>([=] {
          auto native_handle = sycl::get_native<sycl::backend::cuda>(acc);
          is_type_acc[0] = std::is_same_v<decltype(native_handle), c2 *>;
        });
      });
    }

    INFO("Check CUDA kernel function constant buffer accessor interop with \"" +
         typeName + "\" type");
    CHECK(is_type_correct[0]);
  }
};

/** check get_native() returns the correct type for a local accessor
 */
template <>
void test_local_target_accessor<c2>(sycl::queue &queue,
                                    const std::string &typeName) {
  size_t constexpr size = 1;
  bool is_type_correct[size] = {false};
  {
    sycl::buffer<bool> is_type_buff(is_type_correct, sycl::range<1>(size));

    queue.submit([&](sycl::handler &cgh) {
      auto acc =
          sycl::accessor<c2, 1, sycl::access::mode::write, sycl::target::local>(
              sycl::range<1>(size), cgh);
      auto is_type_acc =
          is_type_buff.get_access<sycl::access::mode::write>(cgh);

      cgh.parallel_for_work_group<kernel_local_target_accessor<c2>>(
          sycl::range<1>(1), sycl::range<1>(1), [=](sycl::group<1>) {
            auto native_handle = sycl::get_native<sycl::backend::cuda>(acc);
            is_type_acc[0] = std::is_same_v<decltype(native_handle), c2 *>;
          });
    });
  }

  INFO(
      "Check CUDA kernel function accessor with target::local interop with \"" +
      typeName + "\" type");
  CHECK(is_type_correct[0]);
}

template <typename T>
struct run_all_tests {
  void operator()(sycl::queue &queue, const std::string &typeName) {
    // Check type for accessor
    test_accessor<T> acc_test;
    acc_test(queue, typeName);
    for_type_vectors_marray<test_accessor, T>(queue, typeName);

    // Check type for constant buffer accessor
    test_constant_buffer_accessor<T> const_acc_test;
    const_acc_test(queue, typeName);
    for_type_vectors_marray<test_constant_buffer_accessor, T>(queue, typeName);

    // Check type for local target accessor
    test_local<T> local_acc_test;
    local_acc_test(queue, typeName);
    for_type_vectors_marray<test_local, T>(queue, typeName);
  }
};

template <>
struct run_all_tests<s1> {
  void operator()(sycl::queue &queue, const std::string &typeName) {
    // Check type for accessor
    test_accessor<s1> acc_test;
    acc_test(queue, typeName);

    // Check type for constant buffer accessor
    test_constant_buffer_accessor<s1> const_acc_test;
    const_acc_test(queue, typeName);

    // Check type for local target accessor
    test_local<s1> local_acc_test;
    local_acc_test(queue, typeName);
  }
};

template <>
struct run_all_tests<c1> {
  void operator()(sycl::queue &queue, const std::string &typeName) {
    // Check type for accessor
    test_accessor<c1> acc_test;
    acc_test(queue, typeName);

    // Check type for constant buffer accessor
    test_constant_buffer_accessor<c1> const_acc_test;
    const_acc_test(queue, typeName);

    // Check type for local target accessor
    test_local<c1> local_acc_test;
    local_acc_test(queue, typeName);
  }
};

template <>
struct run_all_tests<c2> {
  void operator()(sycl::queue &queue, const std::string &typeName) {
    // Check type for accessor
    test_accessor<c2> acc_test;
    acc_test(queue, typeName);

    // Check type for constant buffer accessor
    test_constant_buffer_accessor<c2> const_acc_test;
    const_acc_test(queue, typeName);

    // Check type for local target accessor
    test_local<c2> local_acc_test;
    local_acc_test(queue, typeName);
  }
};

#endif  // CUDA_INTEROP_KERNEL_FUNC_TESTS

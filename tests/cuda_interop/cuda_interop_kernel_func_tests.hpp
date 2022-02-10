/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:  (c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#ifndef CUDA_INTEROP_KERNEL_FUNC_TESTS
#define CUDA_INTEROP_KERNEL_FUNC_TESTS

#include "../common/common.h"
#include "../common/type_coverage.h"

namespace cuda_interop_kernel__ {
using namespace sycl_cts;

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
  static const auto types =
      named_type_pack<char, signed char, unsigned char, short int,
                      unsigned short int, int, unsigned int, long int,
                      unsigned long int, long long int, unsigned long long int,
                      float, bool, std::byte, std::int8_t, std::int16_t,
                      std::int32_t, std::int64_t, std::uint8_t, std::uint16_t,
                      std::uint32_t, std::uint64_t, std::size_t>(
          {"char",
           "signed char",
           "unsigned char",
           "short int",
           "unsigned short int",
           "int",
           "unsigned int",
           "long int",
           "unsigned long int",
           "long long int",
           "unsigned long long int",
           "float",
           "bool",
           "std::byte",
           "std::int8_t",
           "std::int16_t",
           "std::int32_t",
           "std::int64_t",
           "std::uint8_t",
           "std::uint16_t",
           "std::uint32_t",
           "std::uint64_t",
           "std::size_t"});
  return types;
}

/** check get_native() returns the correct type for an accessor
 */
template <typename T>
void test_accessor(sycl::queue &queue, sycl_cts::util::logger &log,
                   const std::string &typeName) {
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

      cgh.single_task([=]() {
        auto native_handle =
            sycl::get_native<sycl::backend::ext_oneapi_cuda>(acc);
        is_type_acc[0] = std::is_same_v<decltype(native_handle), T *>;
      });
    });
  }

  if (!is_type_correct[0]) {
    log.note("Test for CUDA kernel function accessor interop failed for \"" +
             typeName + "\" type");
  }
  assert(is_type_correct[0]);
}

/** check get_native() returns the correct type for a constant-buffer accessor
 */
template <typename T>
void test_constant_buffer_accessor(sycl::queue &queue,
                                   sycl_cts::util::logger &log,
                                   const std::string &typeName) {
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

      cgh.single_task([=]() {
        auto native_handle =
            sycl::get_native<sycl::backend::ext_oneapi_cuda>(acc);
        is_type_acc[0] = std::is_same_v<decltype(native_handle), T *>;
      });
    });
  }

  if (!is_type_correct[0]) {
    log.note(
        "Test for CUDA kernel function constant buffer accessor interop failed "
        "for \"" +
        typeName + "\" type");
  }
  assert(is_type_correct[0]);
}

/** check get_native() returns the correct type for an accessor with
 * target::local
 */
template <typename T>
void test_local_target_accessor(sycl::queue &queue, sycl_cts::util::logger &log,
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

      cgh.single_task([=]() {
        auto native_handle =
            sycl::get_native<sycl::backend::ext_oneapi_cuda>(acc);
        is_type_acc[0] = std::is_same_v<decltype(native_handle), T *>;
      });
    });
  }

  if (!is_type_correct[0]) {
    log.note(
        "Test for CUDA kernel function accessor with target::local interop "
        "failed for \"" +
        typeName + "\" type");
  }
  assert(is_type_correct[0]);
}

/** check get_native() returns the correct type for a local_accessor
 */
template <typename T>
void test_local_accessor(sycl::queue &queue, sycl_cts::util::logger &log,
                         const std::string &typeName) {
  size_t constexpr size = 1;
  bool is_type_correct[size] = {false};
  {
    sycl::buffer<bool> is_type_buff(is_type_correct, sycl::range<1>(size));

    queue.submit([&](sycl::handler &cgh) {
      auto acc = sycl::local_accessor<T, 1>(sycl::range<1>(size), cgh);
      auto is_type_acc =
          is_type_buff.get_access<sycl::access::mode::write>(cgh);

      cgh.single_task([=]() {
        auto native_handle =
            sycl::get_native<sycl::backend::ext_oneapi_cuda>(acc);
        is_type_acc[0] = std::is_same_v<decltype(native_handle), T *>;
      });
    });
  }

  if (!is_type_correct[0]) {
    log.note(
        "Test for CUDA kernel function local_accessor interop failed for \"" +
        typeName + "\" type");
  }
  assert(is_type_correct[0]);
}

/** check get_native() returns the correct type for a local_accessor and
 * target::local
 */
template <typename T>
void test_local(sycl::queue &queue, sycl_cts::util::logger &log,
                const std::string &typeName) {
  test_local_target_accessor<T>(queue, log, typeName);
  test_local_accessor<T>(queue, log, typeName);
}

/** check get_native() returns the correct type for an accessor
 */
template <>
void test_accessor<c2>(sycl::queue &queue, sycl_cts::util::logger &log,
                       const std::string &typeName) {
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

      cgh.single_task([=]() {
        auto native_handle =
            sycl::get_native<sycl::backend::ext_oneapi_cuda>(acc);
        is_type_acc[0] = std::is_same_v<decltype(native_handle), c2 *>;
      });
    });
  }

  if (!is_type_correct[0]) {
    log.note("Test for CUDA kernel function accessor interop failed for \"" +
             typeName + "\" type");
  }
  assert(is_type_correct[0]);
}

/** check get_native() returns the correct type for a constant-buffer accessor
 */
template <>
void test_constant_buffer_accessor<c2>(sycl::queue &queue,
                                       sycl_cts::util::logger &log,
                                       const std::string &typeName) {
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

      cgh.single_task([=]() {
        auto native_handle =
            sycl::get_native<sycl::backend::ext_oneapi_cuda>(acc);
        is_type_acc[0] = std::is_same_v<decltype(native_handle), c2 *>;
      });
    });
  }

  if (!is_type_correct[0]) {
    log.note(
        "Test for CUDA kernel function constant buffer accessor interop failed "
        "for \"" +
        typeName + "\" type");
  }
  assert(is_type_correct[0]);
}

/** check get_native() returns the correct type for a local accessor
 */
template <>
void test_local_target_accessor<c2>(sycl::queue &queue,
                                    sycl_cts::util::logger &log,
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

      cgh.single_task([=]() {
        auto native_handle =
            sycl::get_native<sycl::backend::ext_oneapi_cuda>(acc);
        is_type_acc[0] = std::is_same_v<decltype(native_handle), c2 *>;
      });
    });
  }

  if (!is_type_correct[0]) {
    log.note(
        "Test for CUDA kernel function local accessor interop failed for \"" +
        typeName + "\" type");
  }
  assert(is_type_correct[0]);
}

/** @brief
 *  @tparam
 *  @tparam
 */
template <typename T>
struct run_all_tests {
  void operator()(sycl::queue &queue, sycl_cts::util::logger &log,
                  const std::string &typeName) {
    // Check type for accessor
    test_accessor<T>(queue, log, typeName);
    // Check vector types for accessor
    test_accessor<sycl::vec<T, 1>>(queue, log, typeName);
    test_accessor<sycl::vec<T, 2>>(queue, log, typeName);
    test_accessor<sycl::vec<T, 3>>(queue, log, typeName);
    test_accessor<sycl::vec<T, 4>>(queue, log, typeName);
    test_accessor<sycl::vec<T, 8>>(queue, log, typeName);
    test_accessor<sycl::vec<T, 16>>(queue, log, typeName);
    // Check marray for accessor
    test_accessor<sycl::marray<T, 2>>(queue, log, typeName);
    test_accessor<sycl::marray<T, 5>>(queue, log, typeName);
    test_accessor<sycl::marray<T, 10>>(queue, log, typeName);

    // Check type for constant buffer accessor
    test_constant_buffer_accessor<T>(queue, log, typeName);
    // Check vector types for constant buffer accessor
    test_constant_buffer_accessor<sycl::vec<T, 1>>(queue, log, typeName);
    test_constant_buffer_accessor<sycl::vec<T, 2>>(queue, log, typeName);
    test_constant_buffer_accessor<sycl::vec<T, 3>>(queue, log, typeName);
    test_constant_buffer_accessor<sycl::vec<T, 4>>(queue, log, typeName);
    test_constant_buffer_accessor<sycl::vec<T, 8>>(queue, log, typeName);
    test_constant_buffer_accessor<sycl::vec<T, 16>>(queue, log, typeName);
    // Check marray for constant buffer accessor
    test_constant_buffer_accessor<sycl::marray<T, 2>>(queue, log, typeName);
    test_constant_buffer_accessor<sycl::marray<T, 5>>(queue, log, typeName);
    test_constant_buffer_accessor<sycl::marray<T, 10>>(queue, log, typeName);

    // Check type for local target accessor
    test_local<T>(queue, log, typeName);
    // Check vector types for local target accessor
    test_local<sycl::vec<T, 1>>(queue, log, typeName);
    test_local<sycl::vec<T, 2>>(queue, log, typeName);
    test_local<sycl::vec<T, 3>>(queue, log, typeName);
    test_local<sycl::vec<T, 4>>(queue, log, typeName);
    test_local<sycl::vec<T, 8>>(queue, log, typeName);
    test_local<sycl::vec<T, 16>>(queue, log, typeName);
    // Check marray for local target accessor
    test_local<sycl::marray<T, 2>>(queue, log, typeName);
    test_local<sycl::marray<T, 5>>(queue, log, typeName);
    test_local<sycl::marray<T, 10>>(queue, log, typeName);
  }
};

template <>
struct run_all_tests<bool> {
  void operator()(sycl::queue &queue, sycl_cts::util::logger &log,
                  const std::string &typeName) {
    // Check type for accessor
    test_accessor<bool>(queue, log, typeName);

    // Check type for constant buffer accessor
    test_constant_buffer_accessor<bool>(queue, log, typeName);

    // Check type for local target accessor
    test_local<bool>(queue, log, typeName);
  }
};

template <>
struct run_all_tests<s1> {
  void operator()(sycl::queue &queue, sycl_cts::util::logger &log,
                  const std::string &typeName) {
    // Check type for accessor
    test_accessor<s1>(queue, log, typeName);

    // Check type for constant buffer accessor
    test_constant_buffer_accessor<s1>(queue, log, typeName);

    // Check type for local target accessor
    test_local<s1>(queue, log, typeName);
  }
};

template <>
struct run_all_tests<c1> {
  void operator()(sycl::queue &queue, sycl_cts::util::logger &log,
                  const std::string &typeName) {
    // Check type for accessor
    test_accessor<c1>(queue, log, typeName);

    // Check type for constant buffer accessor
    test_constant_buffer_accessor<c1>(queue, log, typeName);

    // Check type for local target accessor
    test_local<c1>(queue, log, typeName);
  }
};

template <>
struct run_all_tests<c2> {
  void operator()(sycl::queue &queue, sycl_cts::util::logger &log,
                  const std::string &typeName) {
    // Check type for accessor
    test_accessor<c2>(queue, log, typeName);

    // Check type for constant buffer accessor
    test_constant_buffer_accessor<c2>(queue, log, typeName);

    // Check type for local target accessor
    test_local<c2>(queue, log, typeName);
  }
};

}  // namespace cuda_interop_kernel__

#endif  // CUDA_INTEROP_KERNEL_FUNC_TESTS

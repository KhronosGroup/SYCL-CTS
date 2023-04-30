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

#ifndef __SYCLCTS_TESTS_COMMON_SEMANTICS_BY_REFERENCE_H
#define __SYCLCTS_TESTS_COMMON_SEMANTICS_BY_REFERENCE_H

#include "common.h"

#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace common_reference_semantics {

/**
 Tests the copy/move constructible/assignable and destructible traits using
 the standard library. */
struct test_traits {
  static constexpr std::size_t result_count = 5;

  /** Performs the check, can be run on host side or kernel side. */
  template <typename T, typename VectorType>
  static void run(VectorType& vec) {
    std::size_t i = 0;
    vec[i++] = std::is_copy_constructible_v<T>;
    vec[i++] = std::is_copy_assignable_v<T>;
    vec[i++] = std::is_destructible_v<T>;
    vec[i++] = std::is_move_constructible_v<T>;
    vec[i++] = std::is_move_assignable_v<T>;

    assert(result_count == i);
  }

  /** Evaluates the results, must be run on host side. */
  template <typename VectorType>
  static void evaluate(VectorType& vec) {
    std::size_t i = 0;

    INFO("traits");
    UNSCOPED_INFO("is_copy_constructible_v");
    CHECK(vec[i++]);
    UNSCOPED_INFO("is_copy_assignable_v");
    CHECK(vec[i++]);
    UNSCOPED_INFO("is_destructible_v");
    CHECK(vec[i++]);
    UNSCOPED_INFO("is_move_constructible_v");
    CHECK(vec[i++]);
    UNSCOPED_INFO("is_move_assignable_v");
    CHECK(vec[i++]);

    assert(result_count == i);
  }
};

/** Tests copy constructible/assignable using a user-defined storage class. */
struct test_copy {
  static constexpr std::size_t result_count = 2;

  /** Performs the check, can be run on host side or kernel side. */
  template <typename Storage, typename T, typename VectorType>
  static void run(VectorType& vec, const T& t) {
    std::size_t i = 0;

    Storage storage(t);

    // Copy constructible
    T s0(t);
    vec[i++] = storage.check(s0);

    // Copy assignable
    T s1 = t;
    vec[i++] = storage.check(s1);

    assert(result_count == i);
  }

  /** Evaluates the results, must be run on host side. */
  template <typename VectorType>
  static void evaluate(VectorType& vec) {
    std::size_t i = 0;

    INFO("copy");
    UNSCOPED_INFO("constructible");
    CHECK(vec[i++]);
    UNSCOPED_INFO("assignable");
    CHECK(vec[i++]);

    assert(result_count == i);
  }
};

/** Tests move constructible/assignable using a user-defined storage class. */
struct test_move {
  static constexpr std::size_t result_count = 2;

  /**
   Performs the check, can be run on host side or kernel side.
   A non-const instance is required as it is moved in this test, which
   may invalidate the instance. */
  template <typename Storage, typename T, typename VectorType>
  static void run(VectorType& vec, T& t) {
    std::size_t i = 0;

    Storage storage(t);

    // Move constructible
    T s0(std::move(t));
    vec[i++] = storage.check(s0);

    // Move assignable
    T s1 = std::move(s0);  // NB: move s0, not t
    vec[i++] = storage.check(s1);

    assert(result_count == i);
  }

  /** Always run on host side. */
  template <typename VectorType>
  static void evaluate(VectorType& vec) {
    std::size_t i = 0;

    INFO("move");
    UNSCOPED_INFO("constructible");
    CHECK(vec[i++]);
    UNSCOPED_INFO("assignable");
    CHECK(vec[i++]);

    assert(result_count == i);
  }
};

/** Tests equality using the equals-operator. */
struct test_equality {
  static constexpr std::size_t result_count = 5;

  /** Performs the check, can be run on host side or kernel side. */
  template <typename T, typename VectorType>
  static void run(VectorType& vec, const T& t) {
    std::size_t i = 0;

    {  // Equality via copy and symmetry: copy constructor
      T s(t);
      vec[i++] = t == s && s == t;
    }
    {  // Equality via copy and symmetry: copy assignment
      T s = t;
      vec[i++] = t == s && s == t;
    }
    {  // Reflexivity
      vec[i++] = t == t;
    }
    {  // Transitivity: copy constructor
      T s(t);
      T u(s);
      vec[i++] = t == u && u == t;
    }
    {  // Transitivity: copy assignment
      T s = t;
      T u = s;
      vec[i++] = t == u && u == t;
    }

    assert(result_count == i);
  }

  /** Evaluates the results, must be run on host side. */
  template <typename VectorType>
  static void evaluate(VectorType& vec) {
    std::size_t i = 0;

    INFO("equality comparable");
    UNSCOPED_INFO("symmetry: copy constructor");
    CHECK(vec[i++]);
    UNSCOPED_INFO("symmetry: copy assignment");
    CHECK(vec[i++]);
    UNSCOPED_INFO("reflexivity");
    CHECK(vec[i++]);
    UNSCOPED_INFO("transitivity: copy constructor");
    CHECK(vec[i++]);
    UNSCOPED_INFO("transitivity: copy assignment");
    CHECK(vec[i++]);

    assert(result_count == i);
  }
};

/**
 Helper function for comparing to standard library hashes, to make a
 compiler-specific exception. */
template <typename T>
bool hash_equality_helper(const T& t0, const T& t1) {
// FIXME: enable when std::hash specializations for local_accessor and
// host_accessor are implemented link to issue
// https://github.com/intel/llvm/issues/8332
#if defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
  if constexpr (std::is_same_v<
                    T, sycl::host_accessor<int, 1,
                                           sycl::access::mode::read_write>> ||
                std::is_same_v<T, sycl::local_accessor<int, 0>>) {
    return false;
  } else {
    return std::hash<T>{}(t0) == std::hash<T>{}(t1);
  }
#else
  return std::hash<T>{}(t0) == std::hash<T>{}(t1);
#endif
}

/** Tests equality using the hash functionality of the standard library. */
struct test_equality_hash {
  static constexpr std::size_t result_count = 5;

  /** Performs the check, can be run on host side or kernel side. */
  template <typename T, typename VectorType>
  static void run(VectorType& vec, const T& t) {
    std::size_t i = 0;

    {  // Equality via copy and symmetry: copy constructor
      T s(t);
      vec[i++] = hash_equality_helper(t, s);
    }
    {  // Equality via copy and symmetry: copy assignment
      T s = t;
      vec[i++] = hash_equality_helper(t, s);
    }
    {  // Reflexivity
      vec[i++] = hash_equality_helper(t, t);
    }
    {  // Transitivity: copy constructor
      T s(t);
      T u(s);
      vec[i++] = hash_equality_helper(t, u);
    }
    {  // Transitivity: copy assignment
      T s = t;
      T u = s;
      vec[i++] = hash_equality_helper(t, u);
    }

    assert(result_count == i);
  }

  /** Evaluates the results, must be run on host side. */
  template <typename VectorType>
  static void evaluate(VectorType& vec) {
    std::size_t i = 0;

    INFO("equality comparable (hash)");
    UNSCOPED_INFO("symmetry: copy constructor");
    CHECK(vec[i++]);
    UNSCOPED_INFO("symmetry: copy assignment");
    CHECK(vec[i++]);
    UNSCOPED_INFO("reflexivity");
    CHECK(vec[i++]);
    UNSCOPED_INFO("transitivity: copy constructor");
    CHECK(vec[i++]);
    UNSCOPED_INFO("transitivity: copy assignment");
    CHECK(vec[i++]);

    assert(result_count == i);
  }
};

/** Tests inequality using the equals-operator. */
struct test_inequality {
  static constexpr std::size_t result_count = 1;

  /**
   Performs the check, can be run on host side or kernel side.
   To test inequality, a second instance is required, which must be
   non-equal to the first instance. */
  template <typename T, typename VectorType>
  static void run(VectorType& vec, const T& t0, const T& t1) {
    std::size_t i = 0;

    // Symmetry
    vec[i++] = t0 != t1 && t1 != t0;

    assert(result_count == i);
  }

  /** Evaluates the results, must be run on host side. */
  template <typename VectorType>
  static void evaluate(VectorType& vec) {
    std::size_t i = 0;

    INFO("inequality comparable");
    UNSCOPED_INFO("symmetry");
    CHECK(vec[i++]);

    assert(result_count == i);
  }
};

/**
 Helper function for comparing to standard library hashes, to make a
 compiler-specific exception. */
template <typename T>
bool hash_inequality_helper(const T& t0, const T& t1) {
#if defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
  if constexpr (std::is_same_v<
                    T, sycl::host_accessor<int, 1,
                                           sycl::access::mode::read_write>> ||
                std::is_same_v<T, sycl::local_accessor<int, 0>>) {
    return false;
  } else {
    return std::hash<T>{}(t0) != std::hash<T>{}(t1);
  }
#else
  return std::hash<T>{}(t0) != std::hash<T>{}(t1);
#endif
}

/** Tests inequality using the hash functionality of the standard library. */
struct test_inequality_hash {
  static constexpr std::size_t result_count = 1;

  /**
   Performs the check, can be run on host side or kernel side.
   To test inequality, a second instance is required, which must be
   non-equal to the first instance. */
  template <typename T, typename VectorType>
  static void run(VectorType& vec, const T& t0, const T& t1) {
    std::size_t i = 0;

    // Symmetry
    vec[i++] = hash_inequality_helper(t0, t1);

    assert(result_count == i);
  }

  /** Evaluates the results, must be run on host side. */
  template <typename VectorType>
  static void evaluate(VectorType& vec) {
    std::size_t i = 0;

    INFO("inequality comparable (hash)");
    UNSCOPED_INFO("symmetry");
    CHECK(vec[i++]);

    assert(result_count == i);
  }
};

/**
 Tests the common reference semantics:
 traits, equality, hash equality, copy. */
template <typename storage, typename T>
void check_host_impl(const T& t, const std::string& type_name) {
  INFO("checking reference semantics on the host application for type \""
       << type_name << "\"");
  std::size_t result_count =
      test_traits::result_count + test_equality::result_count +
      test_equality_hash::result_count + test_copy::result_count;

  std::vector<int> results(result_count, false);
  auto ptr = results.data();
  test_traits::run<T>(ptr);
  ptr += test_traits::result_count;
  test_equality::run(ptr, t);
  ptr += test_equality::result_count;
  test_equality_hash::run(ptr, t);
  ptr += test_equality_hash::result_count;
  test_copy::run<storage>(ptr, t);
  ptr += test_copy::result_count;
  assert(static_cast<std::ptrdiff_t>(result_count) == ptr - results.data());

  ptr = results.data();
  test_traits::evaluate(ptr);
  ptr += test_traits::result_count;
  test_equality::evaluate(ptr);
  ptr += test_equality::result_count;
  test_equality_hash::evaluate(ptr);
  ptr += test_equality_hash::result_count;
  test_copy::evaluate(ptr);
  ptr += test_copy::result_count;
  assert(static_cast<std::ptrdiff_t>(result_count) == ptr - results.data());
}

/** Tests the common reference semantics: inequality, hash inequality. */
template <typename T>
void check_host_impl_inequality(const T& t0, const T& t1) {
  std::size_t result_count =
      test_inequality::result_count + test_inequality_hash::result_count;

  std::vector<int> results(result_count, false);
  auto ptr = results.data();
  test_inequality::run(ptr, t0, t1);
  ptr += test_inequality::result_count;
  test_inequality_hash::run(ptr, t0, t1);
  ptr += test_inequality_hash::result_count;
  assert(static_cast<std::ptrdiff_t>(result_count) == ptr - results.data());

  ptr = results.data();
  test_inequality::evaluate(ptr);
  ptr += test_inequality::result_count;
  test_inequality_hash::evaluate(ptr);
  ptr += test_inequality_hash::result_count;
  assert(static_cast<std::ptrdiff_t>(result_count) == ptr - results.data());
}

/**
 Tests the common reference semantics: move.
 Instance is passed as non-const, invalidates the instance. */
template <typename storage, typename T>
void check_host_impl_move(T& t) {
  std::size_t result_count = test_move::result_count;
  std::vector<int> results(result_count, false);
  auto ptr = results.data();
  test_move::run<storage>(ptr, t);
  ptr += test_move::result_count;
  assert(static_cast<std::ptrdiff_t>(result_count) == ptr - results.data());

  ptr = results.data();
  test_move::evaluate(ptr);
  ptr += test_copy::result_count;
  assert(static_cast<std::ptrdiff_t>(result_count) == ptr - results.data());
}

/**
 Tests all common reference semantics on the host application,
 for a const instance. Invalidates the instance. */
template <typename storage, typename T>
void check_host(const T& t, const std::string& type_name) {
  // usage in check_host_impl implies that t must be non-const
  check_host_impl<storage>(t, type_name);
}

/**
 Tests all common reference semantics on the host application,
 for a non-const instance. Invalidates the instance.
 Is a super set of the const instance tests. */
template <typename storage, typename T>
void check_host(T& t, const std::string& type_name) {
  check_host_impl<storage>(t, type_name);
  check_host_impl_move<storage>(t);  // last since invalidates instance
}

/**
 Tests all common reference semantics on the host application,
 for a non-const instance and a const second and unequal instance.
 Invalidates the first instance.
 Is a super set of the non-const instance tests. */
template <typename storage, typename T>
void check_host(T& t0, const T& t1, const std::string& type_name) {
  check_host_impl<storage>(t0, type_name);
  check_host_impl_inequality(t0, t1);
  check_host_impl_move<storage>(t0);  // last since invalidates instance
}

template <typename T>
struct kernel_name;

/**
 Tests all common reference semantics in a device function.
 The function \p init_func takes a sycl::handler and returns an instance of
 type \p T. */
template <typename storage, typename T, typename InitFunc>
void check_kernel(InitFunc init_func, const std::string& type_name) {
  INFO("checking reference semantics in kernel function for type \""
       << type_name << "\"");
  std::size_t result_count = test_traits::result_count +
                             test_copy::result_count + test_move::result_count;

  std::vector<int> results(result_count, false);
  {
    sycl::buffer<int> buffer(results.data(), sycl::range<1>{result_count});

    auto queue = sycl_cts::util::get_cts_object::queue();
    queue.submit([&](sycl::handler& cgh) {
      auto accessor = buffer.template get_access<sycl::access_mode::write>(cgh);
      T t = init_func(cgh);
      // use non-simple parallel_for to be able to use local_accessor
      cgh.parallel_for<kernel_name<T>>(
          sycl::nd_range<1>{sycl::range<1>{1}, sycl::range<1>{1}},
          [=](sycl::nd_item<1> nd_item) {
            auto ptr = accessor.begin();
            test_traits::run<T>(ptr);
            ptr += test_traits::result_count;
            test_copy::run<storage>(ptr, t);
            ptr += test_copy::result_count;
            test_move::run<storage>(ptr, t);  // last since invalidates instance
            ptr += test_move::result_count;
            assert(static_cast<std::ptrdiff_t>(result_count) ==
                   ptr - accessor.begin());
          });
    });
  }

  auto ptr = results.data();
  test_traits::evaluate(ptr);
  ptr += test_traits::result_count;
  test_copy::evaluate(ptr);
  ptr += test_copy::result_count;
  test_move::evaluate(ptr);
  ptr += test_move::result_count;
  assert(static_cast<std::ptrdiff_t>(result_count) == ptr - results.data());
}

}  // namespace common_reference_semantics

#endif  // __SYCLCTS_TESTS_COMMON_SEMANTICS_BY_REFERENCE_H

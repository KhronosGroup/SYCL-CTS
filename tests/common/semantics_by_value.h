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

#ifndef __SYCLCTS_TESTS_COMMON_SEMANTICS_BY_VALUE_H
#define __SYCLCTS_TESTS_COMMON_SEMANTICS_BY_VALUE_H

#include "common.h"

#include <array>
#include <string>
#include <type_traits>

namespace common_by_value_semantics {

/**
 * @brief Check equality-comparable operations on the host side
 */
template <typename T>
void check_on_host(sycl_cts::util::logger& log, const T& a,
                   const std::string& testName) {
  /** check for reflexivity
   */
  if (!(a == a)) {
    FAIL(testName
         << " is not equality-comparable (operator== reflexivity failed)");
  } else if (a != a) {
    FAIL(testName
         << " is not equality-comparable (operator!= reflexivity failed)");
  }

  /** check for symmetry
   */
  auto b = a;
  const auto& bReadOnly = b;  // force const-correctness
  if (!(a == bReadOnly)) {
    FAIL(testName
         << " is not equality-comparable (operator==, copy constructor)");
  } else if (!(bReadOnly == a)) {
    FAIL(
        testName << " is not equality-comparable (operator== symmetry failed)");
  } else if (a != bReadOnly) {
    FAIL(testName
         << " is not equality-comparable (operator!=, copy constructor)");
  } else if (bReadOnly != a) {
    FAIL(
        testName << " is not equality-comparable (operator!= symmetry failed)");
  }

  /** check for transitivity
   */
  auto c = b;
  const auto& cReadOnly = c;  // force const-correctness
  if (!(a == cReadOnly)) {
    FAIL(testName
         << " is not equality-comparable (operator== transitivity failed)");
  } else if (a != cReadOnly) {
    FAIL(testName
         << " is not equality-comparable (operator!= transitivity  failed)");
  }
}

/**
 * @brief Check equality-comparable operations on the host side with extra
 *        checks for symmetry
 */
template <typename T>
void check_on_host(sycl_cts::util::logger& log, const T& a, const T& other,
                   const std::string& testName) {
  check_on_host(log, a, testName);

  /** extra checks for symmetry (comparsion with other)
   */
  if (a == other) {
    FAIL(testName
         << " is not equality-comparable (operator==, different value)");
  }
  if (other == a) {
    FAIL(testName << " is not equality-comparable (operator== symmetry "
                     "failed, different value)");
  }
  if (!(a != other)) {
    FAIL(testName
         << " is not equality-comparable (operator!=, different value)");
  }
  if (!(other != a)) {
    FAIL(testName << " is not equality-comparable (operator!= symmetry "
                     "failed, different value)");
  }
}

/**
 * @brief Check equality-comparable operations on the device side
 */
template <typename T>
class on_device_checker {
  /**
   * @brief Provides a safe index for checking an operation
   */
  enum class current_check : size_t {
    reflexivity_equal_self,
    reflexivity_not_equal_self,
    equal_copy,
    equal_copy_symmetry,
    not_equal_copy,
    not_equal_copy_symmetry,
    equal_other,
    equal_other_symmetry,
    not_equal_other,
    not_equal_other_symmetry,
    transitivity_equal,
    transitivity_not_equal,
    SIZE  // This should be last
  };
  using success_array_t = std::array<bool, to_integral(current_check::SIZE)>;

 public:
  template <typename kernelT>
  static void run(sycl_cts::util::logger& log, const std::array<T, 2>& items,
                  const std::string& testName) {
    // Store comparison results from kernel into a success array
    success_array_t success;
    std::fill(std::begin(success), std::end(success), false);
    {
      // Perform comparisons on the passed items on the device side
      sycl::buffer<T> itemBuf(items.data(), sycl::range<1>(items.size()));
      sycl::buffer<bool> successBuf(success.data(),
                                    sycl::range<1>(success.size()));

      auto queue = sycl_cts::util::get_cts_object::queue();
      queue.submit([&](sycl::handler& cgh) {
        auto itemAcc =
            itemBuf.template get_access<sycl::access_mode::read>(cgh);
        auto successAcc = successBuf.get_access<sycl::access_mode::write>(cgh);

        cgh.single_task<kernelT>([=]() {
          const auto& a = itemAcc[0];
          const auto& other = itemAcc[1];

          /** check for reflexivity
           */
          successAcc[to_integral(current_check::reflexivity_equal_self)] =
              (a == a);
          successAcc[to_integral(current_check::reflexivity_not_equal_self)] =
              !(a != a);

          /** check for symmetry
           */
          auto copied = a;
          const auto& b = copied;  // force const-correctness
          successAcc[to_integral(current_check::equal_copy)] = (a == b);
          successAcc[to_integral(current_check::equal_copy_symmetry)] =
              (b == a);
          successAcc[to_integral(current_check::not_equal_copy)] = !(a != b);
          successAcc[to_integral(current_check::not_equal_copy_symmetry)] =
              !(b != a);
          successAcc[to_integral(current_check::equal_other)] = !(a == other);
          successAcc[to_integral(current_check::equal_other_symmetry)] =
              !(other == a);
          successAcc[to_integral(current_check::not_equal_other)] =
              (a != other);
          successAcc[to_integral(current_check::not_equal_other_symmetry)] =
              (other != a);

          /** check for transitivity
           */
          auto copiedTwice = copied;
          const auto& c = copiedTwice;  // force const-correctness
          successAcc[to_integral(current_check::transitivity_equal)] = (c == a);
          successAcc[to_integral(current_check::transitivity_not_equal)] =
              (c != other);
        });
      });
    }

    /** check for reflexivity success
     */
    if (!success[to_integral(current_check::reflexivity_equal_self)]) {
      FAIL(testName
           << " is not equality-comparable (operator== reflexivity failed)");
    }
    if (!success[to_integral(current_check::reflexivity_not_equal_self)]) {
      FAIL(testName
           << " is not equality-comparable (operator!= reflexivity failed)");
    }

    /** check for symmetry success
     */
    if (!success[to_integral(current_check::equal_copy)]) {
      FAIL(testName
           << " is not equality-comparable (operator==, copy constructor)");
    } else if (!success[to_integral(current_check::equal_copy_symmetry)]) {
      FAIL(
          testName << " is not equality-comparable (operator== symmetry failed,"
                   << " copy constructor)");
    }
    if (!success[to_integral(current_check::equal_other)]) {
      FAIL(testName
           << " is not equality-comparable (operator==, different value)");
    } else if (!success[to_integral(current_check::equal_other_symmetry)]) {
      FAIL(
          testName << " is not equality-comparable (operator== symmetry failed,"
                   << " different value)");
    }

    if (!success[to_integral(current_check::not_equal_copy)]) {
      FAIL(testName
           << " is not equality-comparable (operator!=, copy constructor)");
    } else if (!success[to_integral(current_check::not_equal_copy_symmetry)]) {
      FAIL(
          testName << " is not equality-comparable (operator!= symmetry failed,"
                   << " copy constructor)");
    }
    if (!success[to_integral(current_check::not_equal_other)]) {
      FAIL(testName
           << " is not equality-comparable (operator!=, different value)");
    } else if (!success[to_integral(current_check::not_equal_other_symmetry)]) {
      FAIL(
          testName << " is not equality-comparable (operator!= symmetry failed,"
                   << " different value)");
    }

    /** check for transitivity success
     */
    if (!success[to_integral(current_check::transitivity_equal)]) {
      FAIL(testName
           << " is not equality-comparable (operator== transitivity failed)");
    } else if (!success[to_integral(current_check::transitivity_not_equal)]) {
      FAIL(testName << " is not equality-comparable"
                    << " (operator!= transitivity  failed)");
    }
  }
};

}  // namespace common_by_value_semantics

#endif  // __SYCLCTS_TESTS_COMMON_SEMANTICS_BY_VALUE_H

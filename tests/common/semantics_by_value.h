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
 * @brief Provides a safe index for checking an operation
 */
enum class current_check : size_t {
  reflexivity_equal_self,
  reflexivity_not_equal_self,
  equal_copy,
  equal_copy_symmetry,
  not_equal_copy,
  not_equal_copy_symmetry,
  transitivity_equal,
  transitivity_not_equal,
  equal_other,
  equal_other_symmetry,
  not_equal_other,
  not_equal_other_symmetry,
  SIZE  // This should be last
};

static const std::array<std::string, to_integral(current_check::SIZE)>
    error_strings = {
        "not equality-comparable (operator== reflexivity failed)",
        "not equality-comparable (operator!= reflexivity failed)",
        "not equality-comparable (operator==, copy constructor)",
        "not equality-comparable (operator== symmetry failed, copy constructor",
        "not equality-comparable (operator!=, copy constructor)",
        "not equality-comparable (operator!= symmetry failed, copy constructor",
        "not equality-comparable (operator== transitivity failed)",
        "not equality-comparable (operator!= transitivity  failed)",
        "not equality-comparable (operator==, different value)",
        "not equality-comparable (operator== symmetry failed, different value)",
        "not equality-comparable (operator!=, different value)",
        "not equality-comparable (operator!= symmetry failed, different value)",
};

inline std::string get_error_string(int code) { return error_strings[code]; }

template <typename T, typename ResultArr>
void check_equality(const T& a, ResultArr& result) {
  /** check for reflexivity
   */
  result[to_integral(current_check::reflexivity_equal_self)] = (a == a);
  result[to_integral(current_check::reflexivity_not_equal_self)] = !(a != a);

  /** check for symmetry
   */
  auto copied = a;
  const auto& b = copied;  // force const-correctness
  result[to_integral(current_check::equal_copy)] = (a == b);
  result[to_integral(current_check::equal_copy_symmetry)] = (b == a);
  result[to_integral(current_check::not_equal_copy)] = !(a != b);
  result[to_integral(current_check::not_equal_copy_symmetry)] = !(b != a);

  /** check for transitivity
   */
  auto copiedTwice = copied;
  const auto& c = copiedTwice;  // force const-correctness
  result[to_integral(current_check::transitivity_equal)] = (c == a);
  result[to_integral(current_check::transitivity_not_equal)] = !(c != a);
}

template <typename T, typename ResultArr>
void check_equality(const T& a, const T& other, ResultArr& result) {
  check_equality(a, result);
  result[to_integral(current_check::equal_other)] = !(a == other);
  result[to_integral(current_check::equal_other_symmetry)] = !(other == a);
  result[to_integral(current_check::not_equal_other)] = (a != other);
  result[to_integral(current_check::not_equal_other_symmetry)] = (other != a);
}

/**
 * @brief Check equality-comparable operations on the host side
 */
template <typename T>
void check_on_host(sycl_cts::util::logger& log, const T& a,
                   const std::string& testName) {
  bool result[to_integral(current_check::SIZE)];
  check_equality(a, result);
  for (int i = 0; i < to_integral(current_check::equal_other); ++i) {
    if (!result[i]) {
      FAIL(testName << " is " << get_error_string(i));
    }
  }
}

/**
 * @brief Check equality-comparable operations on the host side with extra
 *        checks for symmetry
 */
template <typename T>
void check_on_host(sycl_cts::util::logger& log, const T& a, const T& other,
                   const std::string& testName) {
  bool result[to_integral(current_check::SIZE)];
  check_equality(a, other, result);
  for (int i = 0; i < to_integral(current_check::SIZE); ++i) {
    if (!result[i]) {
      FAIL(testName << " is " << get_error_string(i));
    }
  }
}

/**
 * @brief Check equality-comparable operations on the device side
 */
template <typename T>
class on_device_checker {
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
      queue
          .submit([&](sycl::handler& cgh) {
            auto itemAcc =
                itemBuf.template get_access<sycl::access_mode::read>(cgh);
            auto successAcc =
                successBuf.get_access<sycl::access_mode::write>(cgh);

            cgh.single_task<kernelT>([=]() {
              const auto& a = itemAcc[0];
              const auto& other = itemAcc[1];
              check_equality(a, other, successAcc);
            });
          })
          .wait_and_throw();
    }

    for (int i = 0; i < success.size(); ++i) {
      if (!success[i]) {
        FAIL(testName + " is " + get_error_string(i));
      }
    }
  }
};

}  // namespace common_by_value_semantics

#endif  // __SYCLCTS_TESTS_COMMON_SEMANTICS_BY_VALUE_H

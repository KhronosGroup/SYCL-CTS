/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
// Provides verification for common by-value semantics
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_COMMON_COMMON_BY_VALUE_H
#define __SYCLCTS_TESTS_COMMON_COMMON_BY_VALUE_H

#include "common.h"

#include <array>
#include <string>
#include <type_traits>

namespace {

/**
 * @brief Check equality-comparable operations on the host side
 */
template <typename T>
void check_equality_comparable_generic(sycl_cts::util::logger& log, const T& a,
                                       const std::string& test_name) {
  /** check for reflexivity
   */
  if (!(a == a)) {
    FAIL(log, (test_name +
               " is not equality-comparable (operator== reflexivity failed)")
                  .c_str());
  } else if (a != a) {
    FAIL(log, (test_name +
               " is not equality-comparable (operator!= reflexivity failed)")
                  .c_str());
  }

  /** check for symmetry
   */
  auto b = a;
  const auto& bReadOnly = b; // force const-correctness
  if (!(a == bReadOnly)) {
    FAIL(log, (test_name +
               " is not equality-comparable (operator==, copy constructor)")
                  .c_str());
  } else if (!(bReadOnly == a)) {
    FAIL(log, (test_name +
               " is not equality-comparable (operator== symmetry failed)")
                  .c_str());
  } else if (a != bReadOnly) {
    FAIL(log, (test_name +
               " is not equality-comparable (operator!=, copy constructor)")
                  .c_str());
  } else if (bReadOnly != a) {
    FAIL(log, (test_name +
               " is not equality-comparable (operator!= symmetry failed)")
                  .c_str());
  }

  /** check for transitivity
   */
  auto c = b;
  const auto& cReadOnly = c; // force const-correctness
  if (!(a == cReadOnly)) {
    FAIL(log, (test_name +
               " is not equality-comparable (operator== transitivity failed)")
                  .c_str());
  } else if (a != cReadOnly) {
    FAIL(log, (test_name +
               " is not equality-comparable (operator!= transitivity  failed)")
                  .c_str());
  }
}

/**
 * @brief Check equality-comparable operations on the device side
 */
template <typename T>
class equality_comparable_on_device
{
  /**
   * @brief Provides a safe index for checking an operation
   */
  enum class current_check: size_t {
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
  using success_array_t =
      std::array<bool, to_integral(current_check::SIZE)>;

public:
  template <typename kernelT>
  static void check_on_kernel(sycl_cts::util::logger& log,
                              const std::array<T, 2>& items,
                              const std::string& testName) {
    // Store comparison results from kernel into a success array
    success_array_t success;
    std::fill(std::begin(success), std::end(success), false);
    {
      // Perform comparisons on the passed items on the device side
      cl::sycl::buffer<T> itemBuf(items.data(), cl::sycl::range<1>(items.size()));
      cl::sycl::buffer<bool> successBuf(success.data(),
                                        cl::sycl::range<1>(success.size()));

      auto queue = sycl_cts::util::get_cts_object::queue();
      queue.submit([&](cl::sycl::handler& cgh) {
        auto itemAcc =
            itemBuf.template get_access<cl::sycl::access::mode::read>(cgh);
        auto successAcc =
            successBuf.get_access<cl::sycl::access::mode::write>(cgh);

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
          const auto& b = copied; // force const-correctness
          successAcc[to_integral(current_check::equal_copy)] =
              (a == b);
          successAcc[to_integral(current_check::equal_copy_symmetry)] =
              (b == a);
          successAcc[to_integral(current_check::not_equal_copy)] =
              !(a != b);
          successAcc[to_integral(current_check::not_equal_copy_symmetry)] =
              !(b != a);
          successAcc[to_integral(current_check::equal_other)] =
              !(a == other);
          successAcc[to_integral(current_check::equal_other_symmetry)] =
              !(other == a);
          successAcc[to_integral(current_check::not_equal_other)] =
              (a != other);
          successAcc[to_integral(current_check::not_equal_other_symmetry)] =
              (other != a);

          /** check for transitivity
          */
          auto copiedTwice = copied;
          const auto& c = copiedTwice; // force const-correctness
          successAcc[to_integral(current_check::transitivity_equal)] =
              (c == a);
          successAcc[to_integral(current_check::transitivity_not_equal)] =
              (c != other);
        });
      });
    }

    /** check for reflexivity success
     */
    if (!success[to_integral(current_check::reflexivity_equal_self)]) {
      FAIL(log, (testName +
                 " is not equality-comparable (operator== reflexivity failed)")
                    .c_str());
    }
    if (!success[to_integral(current_check::reflexivity_not_equal_self)]) {
      FAIL(log, (testName +
                 " is not equality-comparable (operator!= reflexivity failed)")
                    .c_str());
    }

    /** check for symmetry success
     */
    if (!success[to_integral(current_check::equal_copy)]) {
      FAIL(log, (testName +
                 " is not equality-comparable (operator==, copy constructor)")
                    .c_str());
    } else if (!success[to_integral(current_check::equal_copy_symmetry)]) {
      FAIL(log, (testName +
                 " is not equality-comparable (operator== symmetry failed," +
                 " copy constructor)")
                    .c_str());
    }
    if (!success[to_integral(current_check::equal_other)]) {
      FAIL(log, (testName +
                 " is not equality-comparable (operator==, different value)")
                    .c_str());
    } else if (!success[to_integral(current_check::equal_other_symmetry)]) {
      FAIL(log, (testName +
                 " is not equality-comparable (operator== symmetry failed," +
                 " different value)")
                    .c_str());
    }

    if (!success[to_integral(current_check::not_equal_copy)]) {
      FAIL(log, (testName +
                 " is not equality-comparable (operator!=, copy constructor)")
                    .c_str());
    } else if (!success[to_integral(current_check::not_equal_copy_symmetry)]) {
      FAIL(log, (testName +
                 " is not equality-comparable (operator!= symmetry failed," +
                 " copy constructor)")
                    .c_str());
    }
    if (!success[to_integral(current_check::not_equal_other)]) {
      FAIL(log, (testName +
                 " is not equality-comparable (operator!=, different value)")
                    .c_str());
    } else if (!success[to_integral(current_check::not_equal_other_symmetry)]) {
      FAIL(log, (testName +
                 " is not equality-comparable (operator!= symmetry failed," +
                 " different value)")
                    .c_str());
    }

    /** check for transitivity success
     */
    if (!success[to_integral(current_check::transitivity_equal)]) {
      FAIL(log, (testName +
                 " is not equality-comparable (operator== transitivity failed)")
                    .c_str());
    } else if (!success[to_integral(current_check::transitivity_not_equal)]) {
      FAIL(log, (testName +
                 " is not equality-comparable (operator!= transitivity  failed)")
                    .c_str());
    }
  }
};

}  // namespace

#endif  // __SYCLCTS_TESTS_COMMON_COMMON_BY_VALUE_H

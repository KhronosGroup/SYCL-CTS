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
//  Provides common code for reduction tests
//
*******************************************************************************/

#ifndef __SYCL_CTS_TEST_REDUCTION_COMMON_H
#define __SYCL_CTS_TEST_REDUCTION_COMMON_H

#include "../common/common.h"
#include "../common/type_coverage.h"
// to use size_t
#include <cstddef>

namespace reduction_common {

constexpr bool with_property{true};
constexpr bool without_property{false};

constexpr size_t number_iterations{10};

constexpr int identity_value{0};

constexpr int init_value_without_property_case{99};

static sycl::range<1> range{number_iterations};
static sycl::nd_range<1> nd_range{range, range};

enum class test_case_type {
  each_work_item = 1,
  each_even_work_item = 2,
  no_one_work_item = 3,
  each_work_item_twice = 4,
};

const auto scalar_types =
    named_type_pack<int, float
#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
                    ,
                    char, signed char, unsigned char, short int,
                    unsigned short int, unsigned int, long int,
                    unsigned long int, long long int, unsigned long long int
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
                    >::generate("int", "float"
#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
                                ,
                                "char", "signed char", "unsigned char",
                                "short int", "unsigned short int",
                                "unsigned int", "long int", "unsigned long int",
                                "long long int", "unsigned long long int"
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
    );

/** @brief Returns expected value for testing
 *  @tparam VariableT The type of the variable with which the test runs
 *  @tparam FunctorT The type of the functor with which the test runs
 *  @tparam BufferT The type of the buffer with which the test runs
 *  @param functor The functor (plus, multiplies, etc.) with which the test runs
 *  @param buffer The buffer to be used in parallel_for() function
 *  @param value_for_initialization The value for initializing output variable
 *  @retval Expected value after test execution
 */
template <test_case_type TestCaseT, typename VariableT, typename FunctorT,
          typename BufferT>
VariableT get_expected_value(FunctorT functor, BufferT& buffer,
                             VariableT value_for_initialization) {
  VariableT expected_value{value_for_initialization};
  sycl::host_accessor buf_accessor{buffer};
  if constexpr (TestCaseT == test_case_type::each_work_item) {
    return std::accumulate(buf_accessor.begin(), buf_accessor.end(),
                           value_for_initialization, functor);
  } else if constexpr (TestCaseT == test_case_type::each_even_work_item) {
    int counter = 1;
    for (auto& buf_element : buf_accessor) {
      if (0 == (counter & 1)) {
        expected_value = functor(expected_value, buf_element);
      }
      ++counter;
    }
    return expected_value;
  } else if constexpr (TestCaseT == test_case_type::no_one_work_item) {
    return expected_value;
  } else if constexpr (TestCaseT == test_case_type::each_work_item_twice) {
    for (auto& buf_element : buf_accessor) {
      expected_value = functor(expected_value, buf_element);
      expected_value = functor(expected_value, buf_element);
    }
    return expected_value;
  }
}

/** @brief This function contained common cases from two get_init_value*
 *         functions using property cases and functor type
 *  @tparam VariableT Variable type from type coverage
 *  @tparam FunctorT The type of the functor with which the test runs
 *  @tparam UsePropertyFlagT UseCombineFlagT std::integral_constant type that
 *          let switch between using and don't using
 *          sycl::property::reduction::initialize_to_identity
 *  @retval Value for initializing
 */
template <typename FunctorT, bool UsePropertyFlagT>
void get_init_value_bool(bool& init_value) {
  if constexpr (UsePropertyFlagT) {
    init_value = !sycl::known_identity<FunctorT, bool>::value;
  } else if constexpr (!UsePropertyFlagT) {
    init_value = sycl::known_identity<FunctorT, bool>::value;
  }
}

/** @brief Initialize value for reductions. If UsePropertyFlagT is true and
 *         sycl::has_known_identity for current functor and variable type is
 *         true then we use value that will not be equal to default reduction
 *         value and this value should be ignored if we don't use property or
 *         sycl::has_known_identity is false then we use common value with
 *         value that will be used in expected value initializations
 *  @tparam VariableT Variable type from type coverage
 *  @tparam FunctorT The type of the functor with which the test runs
 *  @tparam UsePropertyFlagT UseCombineFlagT std::integral_constant type that
 *          let switch between using and don't using
 *          sycl::property::reduction::initialize_to_identity
 *  @retval Value for reduction initializing
 */
template <typename VariableT, typename FunctorT,
          bool UsePropertyFlagT = without_property>
VariableT get_init_value_for_reduction() {
  VariableT init_value{};
  if constexpr (std::is_same_v<VariableT, bool>)
    get_init_value_bool<FunctorT, UsePropertyFlagT>(init_value);
  else if constexpr (UsePropertyFlagT) {
    init_value = 50;
  } else {
    init_value = init_value_without_property_case;
  }
  return init_value;
}

/** @brief Initialize value for calculating expected value. If UsePropertyFlagT
 *         is true and sycl::has_known_identity for current functor and variable
 *         type is true then we use value that will not be equal to default
 *         reduction value and this value should be ignored if we don't use
 *         property or sycl::has_known_identity is false then we will use common
 *         value with value that will be used in value for reductions
 *         initializations
 *  @tparam VariableT Variable type from type coverage
 *  @tparam FunctorT The type of the functor with which the test runs
 *  @tparam UsePropertyFlagT UseCombineFlagT std::integral_constant type that
 *          let switch between using and don't using
 *          sycl::property::reduction::initialize_to_identity
 *  @retval Value for expected value initializing
 */
template <typename VariableT, typename FunctorT, bool UsePropertyFlagT = false>
VariableT get_init_value_for_expected_value() {
  VariableT init_value{};
  if constexpr (UsePropertyFlagT) {
    // case when using reduction with initialize_to_identity
    if constexpr (sycl::has_known_identity<FunctorT, VariableT>::value)
      init_value = sycl::known_identity<FunctorT, VariableT>::value;
    else
      init_value = identity_value;
  } else {
    // Otherwise it should use the same initial value as the reduction.
    init_value = get_init_value_for_reduction<VariableT, FunctorT>();
  }
  return init_value;
}

/** @brief Filling buffer for using it in parallel_for and computing expected
 *         value
 *  @tparam VariableT Variable type from type coverage
 *  @tparam BufferT type sycl::buffer object
 *  @param buffer sycl::buffer object
 */
template <typename VariableT, typename BufferT>
void fill_buffer(BufferT& buffer) {
  sycl::host_accessor buf_accessor{buffer};

  bool value_for_filling_bool_buf{true};
  if (std::is_same_v<VariableT, bool>) {
    std::fill(buf_accessor.begin(), buf_accessor.end(), true);
  } else {
    std::iota(buf_accessor.begin(), buf_accessor.end(), 0);
  }
}

/** @brief Construct new sycl::buffer object and fill it with default values
 *  @tparam VariableT Variable type from type coverage
 *  @retval Constructed and filled buffer
 */
template <typename VariableT>
auto get_buffer() {
  sycl::buffer<VariableT> initial_buf{range};
  fill_buffer<VariableT>(initial_buf);
  return initial_buf;
}

/** @brief Check that current device has sycl::aspect::usm_shared_allocations
 *         aspect
 *  @param queue sycl::queue class object
 *  @retval Bool value that corresponds to the presence of this aspect in the
 *          current device
 */
static inline void check_usm_shared_aspect(sycl::queue& queue) {
  bool has_aspect =
      queue.get_device().has(sycl::aspect::usm_shared_allocations);

  if (!has_aspect) {
    SKIP(
        "Device does not support accessing to unified shared memory "
        "allocation");
  }
}

/** @brief A user-defined functor
 *  @tparam VariableT The type of the variable with which the test runs
 */
template <typename VariableT>
struct op_without_identity {
  VariableT operator()(const VariableT& x, const VariableT& y) const {
    return x + y;
  }
};

/** @brief A user-defined class with several scalar member variables, default
 *         constructor and some overloaded operators.
 */
struct custom_type {
  int m_int_field{};
  char m_char_field{};

  custom_type() = default;

  friend bool operator==(const custom_type& c_t_l, const custom_type& c_t_r) {
    return c_t_l.m_int_field == c_t_r.m_int_field &&
           c_t_l.m_char_field == c_t_r.m_char_field;
  }

  operator int() const { return m_int_field; }

  void operator=(int value) {
    m_int_field = value;
    m_char_field = value;
  }

  friend custom_type operator+(const custom_type& c_t_l,
                               const custom_type& c_t_r) {
    custom_type temp;
    temp.m_int_field = c_t_l.m_int_field + c_t_r.m_int_field;
    temp.m_char_field = c_t_l.m_char_field + c_t_r.m_char_field;
    return temp;
  }
};

}  // namespace reduction_common

#endif  // __SYCL_CTS_TEST_REDUCTION_COMMON_H

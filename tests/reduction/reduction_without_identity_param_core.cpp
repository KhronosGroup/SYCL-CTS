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
//  Provides tests for interaction reductions with scalar and bool variables
//  types without identity param.
//
*******************************************************************************/

#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

// FIXME: re-enable when sycl::reduction is implemented in hipSYCL
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL
#include "reduction_without_identity_param_common.h"

namespace reduction_without_identity_param_core {

using namespace reduction_without_identity_param_common;
using namespace reduction_common;

/** @brief Run test for bool type variable
 *  @tparam UsePropertyFlagT UseCombineFlagT std::integral_constant type that
 *          let switch between using and don't using
 *          sycl::property::reduction::initialize_to_identity
 *  @tparam RangeT sycl::range or sycl::nd_range type
 *  @param range sycl::range or sycl::nd_range type object
 *  @param queue sycl::queue class object
 */
template <typename UsePropertyFlagT, typename RangeT>
void run_test_for_bool_variable(RangeT& range, sycl::queue& queue) {
  run_test_for_all_reductions_types<bool, reduction_get_lambda::with_combine,
                                    UsePropertyFlagT::value,
                                    test_case_type::each_work_item>(
      sycl::logical_and<bool>(), range, queue, "bool");

  run_test_for_all_reductions_types<bool, reduction_get_lambda::with_combine,
                                    UsePropertyFlagT::value,
                                    test_case_type::each_work_item>(
      sycl::logical_or<bool>(), range, queue, "bool");
}

/** @brief Run test for reduction that receive sycl::buffer for construct
 *         reducer object
 *  @tparam UsePropertyFlagT UseCombineFlagT std::integral_constant type that
 *          let switch between using and don't using
 *          sycl::property::reduction::initialize_to_identity
 *  @tparam RangeT sycl::range or sycl::nd_range type
 *  @param range sycl::range or sycl::nd_range type object
 *  @param queue sycl::queue class object
 */
template <typename UsePropertyFlagT, typename RangeT>
void run_all_core_tests(RangeT& range, sycl::queue& queue) {
  for_all_types<run_tests_for_all_functors, UsePropertyFlagT>(scalar_types,
                                                              range, queue);

  // sycl::property::reduction::initialize_to_identity cannot be used with
  // reductions that have neither a specified nor known identity.
  if constexpr (!UsePropertyFlagT::value) {
    static_assert(
        !sycl::has_known_identity_v<sycl::plus<custom_type>, custom_type>,
        "sycl::plus<custom_type> should not have a known identity.");
    static_assert(
        !sycl::has_known_identity_v<op_without_identity<int>(), int>,
        "op_without_identity<int>() should not have a known identity.");

    run_test_for_all_reductions_types<
        custom_type, reduction_get_lambda::with_combine,
        UsePropertyFlagT::value, test_case_type::each_work_item>(
        sycl::plus<custom_type>(), range, queue,
        "reduction_common::custom_type");
    run_test_for_all_reductions_types<
        custom_type, reduction_get_lambda::without_combine,
        UsePropertyFlagT::value, test_case_type::each_work_item>(
        sycl::plus<custom_type>(), range, queue,
        "reduction_common::custom_type");
    run_test_for_all_reductions_types<int, reduction_get_lambda::with_combine,
                                      UsePropertyFlagT::value,
                                      test_case_type::each_work_item>(
        op_without_identity<int>(), range, queue, "int with custom functor");
  }

  run_test_for_bool_variable<UsePropertyFlagT>(range, queue);
}

/** @brief Run tests for core types with chosen identity type
 *  @tparam UsePropertyFlagT UseCombineFlagT std::integral_constant type that
 *          let switch between using and don't using
 *          sycl::property::reduction::initialize_to_identity
 *  @param queue sycl::queue class object
 */
template <typename UsePropertyFlagT>
void run_tests_for_identity_type(sycl::queue& queue) {
  run_all_core_tests<UsePropertyFlagT>(range, queue);
  run_all_core_tests<UsePropertyFlagT>(nd_range, queue);
}
}  // namespace reduction_without_identity_param_core

#endif  // !SYCL_CTS_COMPILING_WITH_HIPSYCL

namespace reduction_without_identity_param_core {

// FIXME: re-enable when compilation failure for reduction with custom type is
// fixed and sycl::reduction is implemented in hipSYCL
DISABLED_FOR_TEST_CASE(hipSYCL)
("reduction_without_identity_param_core", "[reduction]")({
  auto queue = sycl_cts::util::get_cts_object::queue();

  run_tests_for_identity_type<run_test_without_property>(queue);
  run_tests_for_identity_type<run_test_with_property>(queue);
});

}  // namespace reduction_without_identity_param_core

/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for interaction reductions with scalar and bool variables
//  types without identity param.
//
*******************************************************************************/

#include "../common/disabled_for_test_case.h"
#include "reduction_without_identity_param_common.h"

namespace reduction_without_identity_param_core {
using namespace sycl_cts;
using namespace reduction_without_identity;
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
                                    UsePropertyFlagT::value>(
      sycl::logical_and<bool>(), range, queue, "bool");

  run_test_for_all_reductions_types<bool, reduction_get_lambda::with_combine,
                                    UsePropertyFlagT::value>(
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

  run_test_for_all_reductions_types<
      custom_type, reduction_get_lambda::with_combine, UsePropertyFlagT::value>(
      sycl::plus<custom_type>(), range, queue, "reduction_common::custom_type");
  run_test_for_all_reductions_types<custom_type,
                                    reduction_get_lambda::without_combine,
                                    UsePropertyFlagT::value>(
      sycl::plus<custom_type>(), range, queue, "reduction_common::custom_type");
  run_test_for_all_reductions_types<int, reduction_get_lambda::with_combine,
                                    UsePropertyFlagT::value>(
      op_without_identity<int>(), range, queue, "int with custom functor");

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

// FIXME: re-enable when compilation failure for reduction with custom type is
// fixed.
DISABLED_FOR_TEST_CASE(DPCPP)
("reduction_without_identity_param_core", "[reduction]")({
  auto queue = util::get_cts_object::queue();

  run_tests_for_identity_type<run_test_without_property>(queue);
  run_tests_for_identity_type<run_test_with_property>(queue);
});

}  // namespace reduction_without_identity_param_core

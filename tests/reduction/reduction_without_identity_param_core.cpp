/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for interaction reductions with scalar and bool variables
//  types without identity param.
//
*******************************************************************************/

#include "reduction_without_identity_param_common.h"

#define TEST_NAME reduction_without_identity_param_core

namespace TEST_NAMESPACE {
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
 *  @param log sycl_cts::util::logger class object
 */
template <typename UsePropertyFlagT, typename RangeT>
void run_test_for_bool_variable(RangeT& range, sycl::queue& queue,
                                sycl_cts::util::logger& log) {
  run_test_for_all_reductions_types<bool, reduction_get_lambda::with_combine,
                                    UsePropertyFlagT>(
      sycl::logical_and<bool>(), range, queue, log, "bool");

  run_test_for_all_reductions_types<bool, reduction_get_lambda::with_combine,
                                    UsePropertyFlagT>(
      sycl::logical_or<bool>(), range, queue, log, "bool");
}

/** @brief Run test for reduction that receive sycl::buffer for construct
 *         reducer object
 *  @tparam UsePropertyFlagT UseCombineFlagT std::integral_constant type that
 *          let switch between using and don't using
 *          sycl::property::reduction::initialize_to_identity
 *  @tparam RangeT sycl::range or sycl::nd_range type
 *  @param range sycl::range or sycl::nd_range type object
 *  @param queue sycl::queue class object
 *  @param log sycl_cts::util::logger class object
 */
template <typename UsePropertyFlagT, typename RangeT>
void run_all_core_tests(RangeT& range, sycl::queue& queue,
                        sycl_cts::util::logger& log) {
  for_all_types<run_tests_for_all_functors, UsePropertyFlagT>(
      scalar_types, range, queue, log);

  run_test_for_all_reductions_types<
      custom_type, reduction_get_lambda::with_combine, UsePropertyFlagT>(
      sycl::plus<custom_type>(), range, queue, log,
      "reduction_common::custom_type");
  run_test_for_all_reductions_types<
      custom_type, reduction_get_lambda::without_combine, UsePropertyFlagT>(
      sycl::plus<custom_type>(), range, queue, log,
      "reduction_common::custom_type");
  run_test_for_all_reductions_types<int, reduction_get_lambda::with_combine,
                                    UsePropertyFlagT>(
      op_without_identity<int>(), range, queue, log, "int with custom functor");

  run_test_for_bool_variable<UsePropertyFlagT>(range, queue, log);
}

/** @brief Run tests for core types with chosen identity type
 *  @tparam UsePropertyFlagT UseCombineFlagT std::integral_constant type that
 *          let switch between using and don't using
 *          sycl::property::reduction::initialize_to_identity
 *  @param queue sycl::queue class object
 *  @param log sycl_cts::util::logger class object
 */
template <typename UsePropertyFlagT>
void run_tests_for_identity_type(sycl::queue& queue,
                                 sycl_cts::util::logger& log) {
  run_all_core_tests<UsePropertyFlagT>(range, queue, log);
  run_all_core_tests<UsePropertyFlagT>(nd_range, queue, log);
}

/** Test instance
 */
class TEST_NAME : public sycl_cts::util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info& out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  void run(util::logger& log) override {
    auto queue = util::get_cts_object::queue();
    run_tests_for_identity_type<without_property>(queue, log);
    run_tests_for_identity_type<with_property>(queue, log);
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;
}  // namespace TEST_NAMESPACE

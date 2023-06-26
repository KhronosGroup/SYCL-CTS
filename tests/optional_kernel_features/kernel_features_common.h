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

//  Provides common functions for optional kernel features tests

#ifndef SYCL_CTS_TEST_KERNEL_FEATURES_COMMON_H
#define SYCL_CTS_TEST_KERNEL_FEATURES_COMMON_H
#include "../../util/sycl_exceptions.h"
#include "../common/common.h"
#include "catch2/matchers/catch_matchers.hpp"
namespace kernel_features_common {
// FIXME: re-enable compilation with hipSYCL or computecpp when `sycl::errc` is supported
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL && !SYCL_CTS_COMPILING_WITH_COMPUTECPP

enum class call_attribute_type {
  external_decorated,
  external_decorated_with_attr,
  non_decorated,
  decorated,
  dummy_decorated,
  dummy_non_decorated,
  type_used,
  type_used_with_attr
};

#ifdef SYCL_EXTERNAL
/**
 * @brief The external function that use T and decorated with attribute
 *
 * @tparam T The type of variable that will use inside the function
 * @tparam aspect Instance of sycl::aspect that will be used in attribute
 */
template <typename T, sycl::aspect aspect>
[[sycl::device_has(aspect)]] SYCL_EXTERNAL void
use_feature_function_external_decorated(const sycl::accessor<bool, 1> &acc);
#endif

/**
 * @brief The function that use T
 *
 * @tparam T The type of variable that will use inside the function
 */
template <typename T>
inline void use_feature_function_non_decorated() {
  unsigned long long temp = 42;
  T feature(temp);
  feature += 42;
}

/**
 * @brief The function that use T
 *
 * @tparam T The type of variable that will use inside the function
 */
template <typename T>
inline void use_feature_function_non_decorated_with_accessor(
    const sycl::accessor<bool, 1> &acc) {
  unsigned long long temp = 42;
  T feature1(temp);
  T feature2(temp);
  feature1 += 42;
  acc[0] = (feature1 == feature2);
}

/**
 * @brief The function that use T and decorated with attribute
 *
 * @tparam T The type of variable that will use inside the function
 * @tparam aspect Instance of sycl::aspect that will be used in attribute
 */
template <typename T, sycl::aspect aspect>
[[sycl::device_has(aspect)]] void use_feature_function_decorated(
    const sycl::accessor<bool, 1> &acc) {
  unsigned long long temp = 42;
  T feature1(temp);
  T feature2(temp);
  feature1 += 42;
  acc[0] = (feature1 == feature2);
}

/**
 * @brief The dummy function that don't use any feature inside and decorated
 * with attribute
 *
 * @tparam aspect Instance of sycl::aspect that will be used in attribute
 */
template <sycl::aspect aspect>
[[sycl::device_has(aspect)]] void dummy_function_decorated(
    const sycl::accessor<bool, 1> &acc) {
  int var1 = 0;
  int var2 = 0;
  var1 += 42;
  acc[0] = (var1 == var2);
}

/**
 * @brief The dummy function that don't use any feature inside
 *
 * @tparam T The type of variable that will use inside the function
 * @tparam aspect Instance of sycl::aspect that will be used in attribute
 */
inline void dummy_function_non_decorated(const sycl::accessor<bool, 1> &acc) {
  int var1 = 0;
  int var2 = 0;
  var1 += 42;
  acc[0] = (var1 == var2);
}

/**
 * @brief Macro for generating code that will use TYPE
 */
#define USE_FEATURE(TYPE)       \
  unsigned long long temp = 42; \
  TYPE feature1(temp);          \
  TYPE feature2(temp);          \
  feature1 += 42;               \
  acc[0] = (feature1 == feature2);

/**
 * @brief Not decorated functor that use feature defined in FeatureTypeT
 */
template <typename FeatureTypeT>
class non_decorated_call_use_feature {
 public:
  sycl::accessor<bool, 1> acc;
  non_decorated_call_use_feature(sycl::accessor<bool, 1> _acc) : acc(_acc) {}
  void operator()() const { USE_FEATURE(FeatureTypeT); }
  void operator()(sycl::item<1>) const { USE_FEATURE(FeatureTypeT); }
  void operator()(sycl::group<1>) const { USE_FEATURE(FeatureTypeT); }
};

/**
 * @brief Not decorated functor that invokes not decorated function that use
 * feature defined in FeatureTypeT
 */
template <typename FeatureTypeT>
class non_decorated_call_non_decorated_function {
 public:
  void operator()() const {
    use_feature_function_non_decorated<FeatureTypeT>();
  }
  void operator()(sycl::item<1>) const {
    use_feature_function_non_decorated<FeatureTypeT>();
  }
  void operator()(sycl::group<1>) const {
    use_feature_function_non_decorated<FeatureTypeT>();
  }
};

/**
 * @brief Non-decorated functor that invokes non-decorated function that uses
 * feature defined in FeatureTypeT
 */
template <typename FeatureTypeT>
class non_decorated_call_non_decorated_function_with_accessor {
 public:
  sycl::accessor<bool, 1> acc;
  non_decorated_call_non_decorated_function_with_accessor(
      sycl::accessor<bool, 1> _acc)
      : acc(_acc) {}
  void operator()() const {
    use_feature_function_non_decorated_with_accessor<FeatureTypeT>(acc);
  }
  void operator()(sycl::item<1>) const {
    use_feature_function_non_decorated_with_accessor<FeatureTypeT>(acc);
  }
  void operator()(sycl::group<1>) const {
    use_feature_function_non_decorated_with_accessor<FeatureTypeT>(acc);
  }
};

/**
 * @brief Not decorated functor that invokes decorated with FeatureAspectT
 * function that use feature defined in FeatureTypeT
 */
template <typename FeatureTypeT, sycl::aspect FeatureAspectT>
class non_decorated_call_decorated_function {
 public:
  sycl::accessor<bool, 1> acc;
  non_decorated_call_decorated_function(sycl::accessor<bool, 1> _acc)
      : acc(_acc) {}
  void operator()() const {
    use_feature_function_decorated<FeatureTypeT, FeatureAspectT>(acc);
  }
  void operator()(sycl::item<1>) const {
    use_feature_function_decorated<FeatureTypeT, FeatureAspectT>(acc);
  }
  void operator()(sycl::group<1>) const {
    use_feature_function_decorated<FeatureTypeT, FeatureAspectT>(acc);
  }
};

#ifdef SYCL_EXTERNAL
/**
 * @brief Not decorated functor that invokes decorated with FeatureAspectT
 * external function that use feature defined in FeatureTypeT
 */
template <typename FeatureTypeT, sycl::aspect FeatureAspectT>
class non_decorated_call_decorated_external_function {
 public:
  sycl::accessor<bool, 1> acc;
  non_decorated_call_decorated_external_function(sycl::accessor<bool, 1> _acc)
      : acc(_acc) {}
  void operator()() const {
    use_feature_function_external_decorated<FeatureTypeT, FeatureAspectT>(acc);
  }
  void operator()(sycl::item<1>) const {
    use_feature_function_external_decorated<FeatureTypeT, FeatureAspectT>(acc);
  }
  void operator()(sycl::group<1>) const {
    use_feature_function_external_decorated<FeatureTypeT, FeatureAspectT>(acc);
  }
};
#endif

/**
 * @brief Not decorated functor that invokes decorated with FeatureAspectT
 * function that don't use any feature
 */
template <sycl::aspect FeatureAspectT>
class non_decorated_call_decorated_dummy {
 public:
  sycl::accessor<bool, 1> acc;
  non_decorated_call_decorated_dummy(sycl::accessor<bool, 1> _acc)
      : acc(_acc) {}
  void operator()() const { dummy_function_decorated<FeatureAspectT>(acc); }
  void operator()(sycl::item<1>) const {
    dummy_function_decorated<FeatureAspectT>(acc);
  }
  void operator()(sycl::group<1>) const {
    dummy_function_decorated<FeatureAspectT>(acc);
  }
};

/**
 * @brief Decorated with FeatureAspectT functor that use feature defined in
 * FeatureTypeT
 */
template <typename FeatureTypeT, sycl::aspect FeatureAspectT>
class decorated_call_use_feature {
 public:
  sycl::accessor<bool, 1> acc;
  decorated_call_use_feature(sycl::accessor<bool, 1> _acc) : acc(_acc) {}
  [[sycl::device_has(FeatureAspectT)]] void operator()() const {
    USE_FEATURE(FeatureTypeT);
  }
  [[sycl::device_has(FeatureAspectT)]] void operator()(sycl::item<1>) const {
    USE_FEATURE(FeatureTypeT);
  }
  [[sycl::device_has(FeatureAspectT)]] void operator()(sycl::group<1>) const {
    USE_FEATURE(FeatureTypeT);
  }
};

#ifdef SYCL_EXTERNAL
/**
 * @brief Decorated with KernelAspectT functor that invokes decorated with
 * FeatureAspectT external function that use feature defined in FeatureTypeT
 */
template <typename FeatureTypeT, sycl::aspect KernelAspectT,
          sycl::aspect FunctionAspectT = KernelAspectT>
class decorated_call_decorated_external_function {
 public:
  sycl::accessor<bool, 1> acc;
  decorated_call_decorated_external_function(sycl::accessor<bool, 1> _acc)
      : acc(_acc) {}
  [[sycl::device_has(KernelAspectT)]] void operator()() const {
    use_feature_function_external_decorated<FeatureTypeT, FunctionAspectT>(acc);
  }
  [[sycl::device_has(KernelAspectT)]] void operator()(sycl::item<1>) const {
    use_feature_function_external_decorated<FeatureTypeT, FunctionAspectT>(acc);
  }
  [[sycl::device_has(KernelAspectT)]] void operator()(sycl::group<1>) const {
    use_feature_function_external_decorated<FeatureTypeT, FunctionAspectT>(acc);
  }
};
#endif

/**
 * @brief Decorated with FeatureAspectT functor that invokes not decorated
 * function that don't use any feature
 */
template <sycl::aspect FeatureAspectT>
class decorated_call_non_decorated_dummy {
 public:
  sycl::accessor<bool, 1> acc;
  decorated_call_non_decorated_dummy(sycl::accessor<bool, 1> _acc)
      : acc(_acc) {}
  [[sycl::device_has(FeatureAspectT)]] void operator()() const {
    dummy_function_non_decorated(acc);
  }
  [[sycl::device_has(FeatureAspectT)]] void operator()(sycl::item<1>) const {
    dummy_function_non_decorated(acc);
  }
  [[sycl::device_has(FeatureAspectT)]] void operator()(sycl::group<1>) const {
    dummy_function_non_decorated(acc);
  }
};

/**
 * @brief The function helps to get another aspect from given aspect
 *
 * @tparam aspect The current aspect based on which needs to get the another
 * object
 * @return Returns sycl::aspect instance that is not equal to given
 */
template <sycl::aspect aspect>
constexpr sycl::aspect get_another_aspect() {
  if constexpr (aspect == sycl::aspect::fp16) {
    return sycl::aspect::fp64;
  } else if constexpr (aspect == sycl::aspect::fp64) {
    return sycl::aspect::atomic64;
  } else {
    return sycl::aspect::fp16;
  }
}

template <typename FeatureTypeT, sycl::aspect FeatureAspectT,
          call_attribute_type CallType>
const auto get_lambda_with_no_arg(const sycl::accessor<bool, 1> &acc) {
  static constexpr sycl::aspect AnotherFeatureAspect =
      get_another_aspect<FeatureAspectT>();
  if constexpr (CallType == call_attribute_type::external_decorated) {
    return [=] {
      use_feature_function_external_decorated<FeatureTypeT, FeatureAspectT>(
          acc);
    };
  } else if constexpr (CallType ==
                       call_attribute_type::external_decorated_with_attr) {
    return [acc] [[sycl::device_has(AnotherFeatureAspect)]] {
      use_feature_function_external_decorated<FeatureTypeT, FeatureAspectT>(
          acc);
    };
  } else if constexpr (CallType == call_attribute_type::non_decorated) {
    return [=] {
      use_feature_function_non_decorated_with_accessor<FeatureTypeT>(acc);
    };
  } else if constexpr (CallType == call_attribute_type::decorated) {
    return [=] {
      use_feature_function_decorated<FeatureTypeT, FeatureAspectT>(acc);
    };
  } else if constexpr (CallType == call_attribute_type::dummy_decorated) {
    return [=] { dummy_function_decorated<FeatureAspectT>(acc); };
  } else if constexpr (CallType == call_attribute_type::dummy_non_decorated) {
    return [=] [[sycl::device_has(FeatureAspectT)]] {
      dummy_function_non_decorated(acc);
    };
  } else if constexpr (CallType == call_attribute_type::type_used) {
    return [acc] { USE_FEATURE(FeatureTypeT); };
  } else if constexpr (CallType == call_attribute_type::type_used_with_attr) {
    return [=] [[sycl::device_has(AnotherFeatureAspect)]] {
      USE_FEATURE(FeatureTypeT);
    };
  }
}

template <typename FeatureTypeT, sycl::aspect FeatureAspectT,
          call_attribute_type CallType>
const auto get_lambda_with_item_arg(const sycl::accessor<bool, 1> &acc) {
  static constexpr sycl::aspect AnotherFeatureAspect =
      get_another_aspect<FeatureAspectT>();
  if constexpr (CallType == call_attribute_type::external_decorated) {
    return [=](sycl::item<1>) {
      use_feature_function_external_decorated<FeatureTypeT, FeatureAspectT>(
          acc);
    };
  } else if constexpr (CallType ==
                       call_attribute_type::external_decorated_with_attr) {
    return [acc](sycl::item<1>) [[sycl::device_has(AnotherFeatureAspect)]] {
      use_feature_function_external_decorated<FeatureTypeT, FeatureAspectT>(
          acc);
    };
  } else if constexpr (CallType == call_attribute_type::non_decorated) {
    return [=](sycl::item<1>) {
      use_feature_function_non_decorated_with_accessor<FeatureTypeT>(acc);
    };
  } else if constexpr (CallType == call_attribute_type::decorated) {
    return [=](sycl::item<1>) {
      use_feature_function_decorated<FeatureTypeT, FeatureAspectT>(acc);
    };
  } else if constexpr (CallType == call_attribute_type::dummy_decorated) {
    return
        [=](sycl::item<1>) { dummy_function_decorated<FeatureAspectT>(acc); };
  } else if constexpr (CallType == call_attribute_type::dummy_non_decorated) {
    return [=](sycl::item<1>) [[sycl::device_has(FeatureAspectT)]] {
      dummy_function_non_decorated(acc);
    };
  } else if constexpr (CallType == call_attribute_type::type_used) {
    return [=](sycl::item<1>) { USE_FEATURE(FeatureTypeT); };
  } else if constexpr (CallType == call_attribute_type::type_used_with_attr) {
    return [acc](sycl::item<1>) [[sycl::device_has(AnotherFeatureAspect)]] {
      USE_FEATURE(FeatureTypeT);
    };
  }
}
template <typename FeatureTypeT, sycl::aspect FeatureAspectT,
          call_attribute_type CallType>
const auto get_lambda_with_group_arg(const sycl::accessor<bool, 1> &acc) {
  static constexpr sycl::aspect AnotherFeatureAspect =
      get_another_aspect<FeatureAspectT>();
  if constexpr (CallType == call_attribute_type::external_decorated) {
    return [=](sycl::group<1>) {
      use_feature_function_external_decorated<FeatureTypeT, FeatureAspectT>(
          acc);
    };
  } else if constexpr (CallType ==
                       call_attribute_type::external_decorated_with_attr) {
    return [acc](sycl::group<1>) [[sycl::device_has(AnotherFeatureAspect)]] {
      use_feature_function_external_decorated<FeatureTypeT, FeatureAspectT>(
          acc);
    };
  } else if constexpr (CallType == call_attribute_type::non_decorated) {
    return [=](sycl::group<1>) {
      use_feature_function_non_decorated_with_accessor<FeatureTypeT>(acc);
    };
  } else if constexpr (CallType == call_attribute_type::decorated) {
    return [=](sycl::group<1>) {
      use_feature_function_decorated<FeatureTypeT, FeatureAspectT>(acc);
    };
  } else if constexpr (CallType == call_attribute_type::dummy_decorated) {
    return
        [=](sycl::group<1>) { dummy_function_decorated<FeatureAspectT>(acc); };
  } else if constexpr (CallType == call_attribute_type::dummy_non_decorated) {
    return [=](sycl::group<1>) [[sycl::device_has(FeatureAspectT)]] {
      dummy_function_non_decorated(acc);
    };
  } else if constexpr (CallType == call_attribute_type::type_used) {
    return [=](sycl::group<1>) { USE_FEATURE(FeatureTypeT); };
  } else if constexpr (CallType == call_attribute_type::type_used_with_attr) {
    return [acc](sycl::group<1>) [[sycl::device_has(AnotherFeatureAspect)]] {
      USE_FEATURE(FeatureTypeT);
    };
  }
}

/**
 * @brief The function helps to check if async exception was thrown from given
 * queue. The function will fail the test according to the is_exception_expected
 * parameter.
 *
 * @param queue The sycl::queue instance for exception handling
 * @param is_exception_expected The flag shows if the function will expect
 * exception or not
 */
inline void check_async_exception(sycl::queue &queue,
                                  const bool is_exception_expected) {
  INFO("Check if async exception was thrown");
  bool is_async_exception_thrown{};
  try {
    queue.throw_asynchronous();
    is_async_exception_thrown = false;
  } catch (const sycl::exception &e) {
    is_async_exception_thrown = true;
  }
  CHECK(is_async_exception_thrown == is_exception_expected);
}

/**
 * @brief The function helps to execute actions and check if the expected
 * exception was thrown. Actions will be passed to the CHECK_THROWS_MATCHES or
 * CHECK_NOTHROW depending on the flag is_exception_expected.
 *
 * @tparam SingleTaskActionT Type of single_task_action. Can be deduced from the
 * argument.
 * @tparam ParallelForActionT Type of parallel_for_action. Can be deduced from
 * the argument.
 * @tparam ParallelForWgActionT Type of parallel_for_wg_action. Can be deduced
 * from the argument.
 * @param is_exception_expected The flag shows if exception expected from
 * the kernel
 * @param errc_expected The error code that expected from sycl::exception
 * @param queue The sycl::queue instance for device
 * @param description String description of the tasks that will be executed
 * @param single_task_action Task for single_task invocation
 * @param parallel_for_action Task for parallel_for invocation
 * @param parallel_for_wg_action Task for parallel_for_work_group invocation
 */

template <typename SingleTaskActionT, typename ParallelForActionT,
          typename ParallelForWgActionT>
void execute_tasks_and_check_exception(
    const bool is_exception_expected, const sycl::errc errc_expected,
    sycl::queue &queue, const std::string &description,
    SingleTaskActionT single_task_action,
    ParallelForActionT parallel_for_action,
    ParallelForWgActionT parallel_for_wg_action) {
  const std::string single_task_desc =
      "Execution of " + description + " in single_task";
  const std::string parallel_for_desc =
      "Execution of " + description + " in parallel_for";
  const std::string parallel_for_wg_desc =
      "Execution of " + description + " in parallel_for_work_group";

  if (is_exception_expected) {
    {
      INFO(single_task_desc);
      CHECK_THROWS_MATCHES(single_task_action(), sycl::exception,
                           sycl_cts::util::equals_exception(errc_expected));
      check_async_exception(queue, false);
    }
    {
      INFO(parallel_for_desc);
      CHECK_THROWS_MATCHES(parallel_for_action(), sycl::exception,
                           sycl_cts::util::equals_exception(errc_expected));
      check_async_exception(queue, false);
    }
    {
      INFO(parallel_for_wg_desc);
      CHECK_THROWS_MATCHES(parallel_for_wg_action(), sycl::exception,
                           sycl_cts::util::equals_exception(errc_expected));
      check_async_exception(queue, false);
    }
  } else {
    {
      INFO(single_task_desc);
      CHECK_NOTHROW(single_task_action);
      check_async_exception(queue, false);
    }
    {
      INFO(parallel_for_desc);
      CHECK_NOTHROW(parallel_for_action);
      check_async_exception(queue, false);
    }
    {
      INFO(parallel_for_wg_desc);
      CHECK_NOTHROW(parallel_for_wg_action);
      check_async_exception(queue, false);
    }
  }
}

template <typename ParallelForActionT, typename ParallelForWgActionT>
void execute_tasks_and_check_exception(
    const bool is_exception_expected, const sycl::errc errc_expected,
    sycl::queue &queue, const std::string &description,
    ParallelForActionT parallel_for_action,
    ParallelForWgActionT parallel_for_wg_action) {
  const std::string parallel_for_desc =
      "Execution of " + description + " in parallel_for";
  const std::string parallel_for_wg_desc =
      "Execution of " + description + " in parallel_for_work_group";

  if (is_exception_expected) {
    {
      INFO(parallel_for_desc);
      CHECK_THROWS_MATCHES(parallel_for_action(), sycl::exception,
                           sycl_cts::util::equals_exception(errc_expected));
      check_async_exception(queue, false);
    }
    {
      INFO(errc_expected);
      CHECK_THROWS_MATCHES(parallel_for_wg_action(), sycl::exception,
                           sycl_cts::util::equals_exception(errc_expected));
      check_async_exception(queue, false);
    }
  } else {
    {
      INFO(parallel_for_desc);
      CHECK_NOTHROW(parallel_for_action);
      check_async_exception(queue, false);
    }
    {
      INFO(parallel_for_wg_desc);
      CHECK_NOTHROW(parallel_for_wg_action);
      check_async_exception(queue, false);
    }
  }
}

enum class call_type { no_arg, item_arg, group_arg };

template <typename KernelName, call_type CallType>
class kernel_separate_lambda;

/**
 * @brief The function helps to run separate lambdas in the kernel by
 * executing them in single_task, parallel_for and parallel_for_work_group.
 * The function also expects exception depending on is_exception_expected
 * flag.
 *
 * @tparam KernelName The name of the kernel. All \p run_separate_lambda calls
 *         should have a unique name.
 * @tparam LambdaNoArg The type of lambda for single_task invocation
 * @tparam LambdaItemArg The type of lambda for parallel_for invocation
 * @tparam LambdaGroupArg The type of lambda for parallel_for_work_group
 * invocation
 * @param is_exception_expected The flag shows if exception expected from
 * the kernel
 * @param errc_expected The error code that expected from sycl::exception
 * @param queue The sycl::queue instance for device
 * @param separate_lambda_no_arg The lambda for single_task invocation
 * @param separate_lambda_item_arg The lambda for parallel_for invocation
 * @param separate_lambda_group_arg The lambda for parallel_for_work_group
 * invocation
 */
template <typename KernelName, typename LambdaNoArg, typename LambdaItemArg,
          typename LambdaGroupArg>
void run_separate_lambda(const bool is_exception_expected,
                         const sycl::errc errc_expected, sycl::queue &queue,
                         LambdaNoArg separate_lambda_no_arg,
                         LambdaItemArg separate_lambda_item_arg,
                         LambdaGroupArg separate_lambda_group_arg) {
  auto single_task_action = [&queue, separate_lambda_no_arg] {
    queue
        .submit([&](sycl::handler &cgh) {
          cgh.single_task<
              kernel_separate_lambda<KernelName, call_type::no_arg>>(
              separate_lambda_no_arg);
        })
        .wait();
  };
  auto parallel_for_action = [&queue, separate_lambda_item_arg] {
    queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<
              kernel_separate_lambda<KernelName, call_type::item_arg>>(
              sycl::range{1}, separate_lambda_item_arg);
        })
        .wait();
  };
  auto parallel_for_wg_action = [&queue, separate_lambda_group_arg] {
    queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for_work_group<
              kernel_separate_lambda<KernelName, call_type::group_arg>>(
              sycl::range{1}, sycl::range{1}, separate_lambda_group_arg);
        })
        .wait();
  };

  execute_tasks_and_check_exception(
      is_exception_expected, errc_expected, queue, "separate lambda",
      single_task_action, parallel_for_action, parallel_for_wg_action);
}

template <typename KernelName, size_t Size = 1, int Dimensions = 1,
          typename LambdaItemArg, typename LambdaGroupArg>
void run_separate_lambda_nd_range(const bool is_exception_expected,
                                  const sycl::errc errc_expected,
                                  sycl::queue& queue,
                                  LambdaItemArg separate_lambda_nd_item_arg,
                                  LambdaGroupArg separate_lambda_group_arg) {
  auto range =
      sycl_cts::util::get_cts_object::range<Dimensions>::get(Size, Size, Size);
  auto parallel_for_action = [&queue, separate_lambda_nd_item_arg, range] {
    queue
        .submit([&](sycl::handler& cgh) {
          cgh.parallel_for<
              kernel_separate_lambda<KernelName, call_type::item_arg>>(
              sycl::nd_range<Dimensions>{range, range},
              separate_lambda_nd_item_arg);
        })
        .wait();
  };

  auto parallel_for_wg_action = [&queue, separate_lambda_group_arg, range] {
    auto groupRange =
        sycl_cts::util::get_cts_object::range<Dimensions>::get(1, 1, 1);
    queue
        .submit([&](sycl::handler& cgh) {
          cgh.parallel_for_work_group<
              kernel_separate_lambda<KernelName, call_type::group_arg>>(
              groupRange, range, separate_lambda_group_arg);
        })
        .wait();
  };

  execute_tasks_and_check_exception(is_exception_expected, errc_expected, queue,
                                    "separate lambda", parallel_for_action,
                                    parallel_for_wg_action);
}

/**
 * @brief The function helps to run separate lambdas in the kernel by
 * executing them in single_task, parallel_for and parallel_for_work_group.
 * The function also expects exception depending on is_exception_expected
 * flag.
 *
 * @tparam KernelName The name of the kernel. All \p run_separate_lambda calls
 *         should have a unique name.
 * @tparam FeatureTypeT feature type
 * @tparam FeatureAspectT aspect type
 * @tparam CallType type of call_attribute_type used to get right lambda
 * invocation
 * @param is_exception_expected The flag shows if exception expected from
 * the kernel
 * @param errc_expected The error code that expected from sycl::exception
 * @param queue The sycl::queue instance for device
 * invocation
 */
template <typename KernelName, typename FeatureTypeT,
          sycl::aspect Aspect = sycl::aspect::cpu,
          call_attribute_type CallType = call_attribute_type::type_used>
void run_separate_lambda_with_accessor(const bool is_exception_expected,
                                       const sycl::errc errc_expected,
                                       sycl::queue &queue) {
  auto single_task_action = [&queue] {
    bool value = false;
    sycl::buffer<bool, 1> buffer(&value, sycl::range<1>(1));
    queue
        .submit([&](sycl::handler &cgh) {
          auto acc = buffer.get_access(cgh);
          const auto lambda_no_arg =
              get_lambda_with_no_arg<FeatureTypeT, Aspect, CallType>(acc);
          cgh.single_task<
              kernel_separate_lambda<KernelName, call_type::no_arg>>(
              lambda_no_arg);
        })
        .wait();
  };
  auto parallel_for_action = [&queue] {
    bool value = false;
    sycl::buffer<bool, 1> buffer(&value, sycl::range<1>(1));
    queue
        .submit([&](sycl::handler &cgh) {
          auto acc = buffer.get_access(cgh);
          const auto lambda_item_arg =
              get_lambda_with_item_arg<FeatureTypeT, Aspect, CallType>(acc);
          cgh.parallel_for<
              kernel_separate_lambda<KernelName, call_type::item_arg>>(
              sycl::range{1}, lambda_item_arg);
        })
        .wait();
  };
  auto parallel_for_wg_action = [&queue] {
    bool value = false;
    sycl::buffer<bool, 1> buffer(&value, sycl::range<1>(1));
    queue
        .submit([&](sycl::handler &cgh) {
          auto acc = buffer.get_access(cgh);
          const auto lambda_group_arg =
              get_lambda_with_group_arg<FeatureTypeT, Aspect, CallType>(acc);
          cgh.parallel_for_work_group<
              kernel_separate_lambda<KernelName, call_type::group_arg>>(
              sycl::range{1}, sycl::range{1}, lambda_group_arg);
        })
        .wait();
  };

  execute_tasks_and_check_exception(is_exception_expected, errc_expected, queue,
                                    "separate lambda", single_task_action,
                                    parallel_for_action, parallel_for_action);
}

template <typename Functor>
struct kernel_single_task;
template <typename Functor>
struct kernel_parallel_for;
template <typename Functor>
struct kernel_parallel_for_wg;

/**
 * @brief The function helps to run functors in the kernel by executing
 * them in single_task, parallel_for and parallel_for_work_group. The function
 * also expects exception depending on is_exception_expected flag.
 *
 * @tparam Functor The type of functor for single_task,parallel_for, and
 * parallel_for_work_group invocations
 * @param is_exception_expected The flag shows if exception expected from the
 * @param errc_expected The error code that expected from sycl::exception
 * @param queue The sycl::queue instance for device
 * kernel
 */
template <typename Functor>
void run_functor(const bool is_exception_expected,
                 const sycl::errc errc_expected, sycl::queue &queue) {
  auto single_task_action = [&queue] {
    queue
        .submit([&](sycl::handler &cgh) {
          cgh.single_task<kernel_single_task<Functor>>(Functor{});
        })
        .wait();
  };
  auto parallel_for_action = [&queue] {
    queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<kernel_parallel_for<Functor>>(sycl::range{1},
                                                         Functor{});
        })
        .wait();
  };
  auto parallel_for_wg_action = [&queue] {
    queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for_work_group<kernel_parallel_for_wg<Functor>>(
              sycl::range{1}, sycl::range{1}, Functor{});
        })
        .wait();
  };

  execute_tasks_and_check_exception(
      is_exception_expected, errc_expected, queue, "functor",
      single_task_action, parallel_for_action, parallel_for_wg_action);
}

template <typename Functor, size_t Size = 1, int Dimensions = 1>
void run_functor_nd_range(const bool is_exception_expected,
                          const sycl::errc errc_expected, sycl::queue& queue) {
  auto range =
      sycl_cts::util::get_cts_object::range<Dimensions>::get(Size, Size, Size);

  auto parallel_for_action = [&queue, &range] {
    queue
        .submit([&](sycl::handler& cgh) {
          cgh.parallel_for<kernel_parallel_for<Functor>>(
              sycl::nd_range<Dimensions>{range, range}, Functor{});
        })
        .wait();
  };
  auto parallel_for_wg_action = [&queue, &range] {
    auto groupRange =
        sycl_cts::util::get_cts_object::range<Dimensions>::get(1, 1, 1);
    queue
        .submit([&](sycl::handler& cgh) {
          cgh.parallel_for_work_group<kernel_parallel_for_wg<Functor>>(
              groupRange, range, Functor{});
        })
        .wait();
  };

  execute_tasks_and_check_exception(is_exception_expected, errc_expected, queue,
                                    "functor", parallel_for_action,
                                    parallel_for_wg_action);
}

/**
 * @brief The function helps to run functors in the kernel by executing
 * them in single_task, parallel_for and parallel_for_work_group. The function
 * also expects exception depending on is_exception_expected flag.
 *
 * @tparam Functor The type of functor for single_task,parallel_for, and
 * parallel_for_work_group invocations
 * @param is_exception_expected The flag shows if exception expected from the
 * @param errc_expected The error code that expected from sycl::exception
 * @param queue The sycl::queue instance for device
 * kernel
 */
template <typename Functor>
void run_functor_with_accessor(const bool is_exception_expected,
                               const sycl::errc errc_expected,
                               sycl::queue &queue) {
  auto single_task_action = [&queue] {
    bool value = false;
    sycl::buffer<bool, 1> buffer(&value, sycl::range<1>(1));
    queue
        .submit([&](sycl::handler &cgh) {
          auto acc = buffer.get_access(cgh);
          cgh.single_task<kernel_single_task<Functor>>(Functor{acc});
        })
        .wait();
  };
  auto parallel_for_action = [&queue] {
    bool value = false;
    sycl::buffer<bool, 1> buffer(&value, sycl::range<1>(1));
    queue
        .submit([&](sycl::handler &cgh) {
          auto acc = buffer.get_access(cgh);
          cgh.parallel_for<kernel_parallel_for<Functor>>(sycl::range{1},
                                                         Functor{acc});
        })
        .wait();
  };
  auto parallel_for_wg_action = [&queue] {
    bool value = false;
    sycl::buffer<bool, 1> buffer(&value, sycl::range<1>(1));
    queue
        .submit([&](sycl::handler &cgh) {
          auto acc = buffer.get_access(cgh);
          cgh.parallel_for_work_group<kernel_parallel_for_wg<Functor>>(
              sycl::range{1}, sycl::range{1}, Functor{acc});
        })
        .wait();
  };

  execute_tasks_and_check_exception(is_exception_expected, errc_expected, queue,
                                    "functor", single_task_action,
                                    parallel_for_action, parallel_for_action);
}

#define NO_ATTRIBUTE   /*no attribute*/
#define NO_KERNEL_BODY /*no kernel code*/

template <typename KernelName, call_type CallType>
class kernel_submission_call;

/**
 * @brief The function like macros that helps to define and run kernels through
 * lambda in submission call. Macro generates execution in single_task,
 * parallel_for and parallel_for_work_group. The function
 * also expects exception depending on IS_EXCEPTION_EXPECTED flag.
 *
 * @param IS_EXCEPTION_EXPECTED The flag shows if exception expected from the
 * kernel
 * @param ERRC The error code that expected from sycl::exception
 * @param QUEUE The sycl::queue instance for device
 * @param ATTRIBUTE The attribute that will be applied to the submission call.
 * @param KERNEL_NAME Name of the kernel. Each \p RUN_SUBMISSION_CALL call
 *        should have a unique name,
 * @param __VA_ARGS__ Body of the submission call that have to be executed on
 * the device
 */
#define RUN_SUBMISSION_CALL(IS_EXCEPTION_EXPECTED, ERRC, QUEUE, ATTRIBUTE,  \
                            KERNEL_NAME, ...)                               \
                                                                            \
  {                                                                         \
    auto single_task_action = [&QUEUE] {                                    \
      bool value = false;                                                   \
      sycl::buffer<bool, 1> buffer(&value, sycl::range<1>(1));              \
      QUEUE                                                                 \
          .submit([&](sycl::handler &cgh) {                                 \
            auto acc = buffer.get_access(cgh);                              \
            cgh.single_task<                                                \
                kernel_submission_call<KERNEL_NAME, call_type::no_arg>>(    \
                [=]() ATTRIBUTE { __VA_ARGS__; });                          \
          })                                                                \
          .wait();                                                          \
    };                                                                      \
    auto parallel_for_action = [&QUEUE] {                                   \
      bool value = false;                                                   \
      sycl::buffer<bool, 1> buffer(&value, sycl::range<1>(1));              \
      QUEUE                                                                 \
          .submit([&](sycl::handler &cgh) {                                 \
            auto acc = buffer.get_access(cgh);                              \
            cgh.parallel_for<                                               \
                kernel_submission_call<KERNEL_NAME, call_type::item_arg>>(  \
                sycl::range{1},                                             \
                [=](sycl::item<1>) ATTRIBUTE { __VA_ARGS__; });             \
          })                                                                \
          .wait();                                                          \
    };                                                                      \
    auto parallel_for_wg_action = [&QUEUE] {                                \
      bool value = false;                                                   \
      sycl::buffer<bool, 1> buffer(&value, sycl::range<1>(1));              \
      QUEUE                                                                 \
          .submit([&](sycl::handler &cgh) {                                 \
            auto acc = buffer.get_access(cgh);                              \
            cgh.parallel_for_work_group<                                    \
                kernel_submission_call<KERNEL_NAME, call_type::group_arg>>( \
                sycl::range{1}, sycl::range{1},                             \
                [=](sycl::group<1>) ATTRIBUTE { __VA_ARGS__; });            \
          })                                                                \
          .wait();                                                          \
    };                                                                      \
                                                                            \
    execute_tasks_and_check_exception(                                      \
        IS_EXCEPTION_EXPECTED, ERRC, QUEUE, "submission call",              \
        single_task_action, parallel_for_action, parallel_for_wg_action);   \
  }

#define RUN_SUBMISSION_CALL_ND_RANGE(SIZE, D, IS_EXCEPTION_EXPECTED, ERRC,    \
                                     QUEUE, ATTRIBUTE, KERNEL_NAME, ...)      \
                                                                              \
  {                                                                           \
    auto parallel_for_action = [&QUEUE] {                                     \
      auto range =                                                            \
          sycl_cts::util::get_cts_object::range<D>::get(SIZE, SIZE, SIZE);    \
      QUEUE                                                                   \
          .submit([&](sycl::handler& cgh) {                                   \
            cgh.parallel_for<                                                 \
                kernel_submission_call<KERNEL_NAME, call_type::item_arg>>(    \
                sycl::nd_range<D>{range, range},                              \
                [=](sycl::nd_item<D>) ATTRIBUTE { __VA_ARGS__; });            \
          })                                                                  \
          .wait();                                                            \
    };                                                                        \
    auto parallel_for_wg_action = [&QUEUE] {                                  \
      auto range =                                                            \
          sycl_cts::util::get_cts_object::range<D>::get(SIZE, SIZE, SIZE);    \
      auto groupRange =                                                       \
          sycl_cts::util::get_cts_object::range<D>::get(1, 1, 1);             \
      QUEUE                                                                   \
          .submit([&](sycl::handler& cgh) {                                   \
            cgh.parallel_for_work_group<                                      \
                kernel_submission_call<KERNEL_NAME, call_type::group_arg>>(   \
                groupRange, range,                                            \
                [=](sycl::group<D>) ATTRIBUTE { __VA_ARGS__; });              \
          })                                                                  \
          .wait();                                                            \
    };                                                                        \
                                                                              \
    execute_tasks_and_check_exception(IS_EXCEPTION_EXPECTED, ERRC, QUEUE,     \
                                      "submission call", parallel_for_action, \
                                      parallel_for_wg_action);                \
  }

#endif  // #if !SYCL_CTS_COMPILING_WITH_HIPSYCL &&
        // !SYCL_CTS_COMPILING_WITH_COMPUTECPP
}  // namespace kernel_features_common

#endif  // SYCL_CTS_TEST_KERNEL_FEATURES_COMMON_H

/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides common functions for optional kernel features tests
//
*******************************************************************************/

#ifndef SYCL_CTS_TEST_KERNEL_FEATURES_COMMON_H
#define SYCL_CTS_TEST_KERNEL_FEATURES_COMMON_H
#include "../common/common.h"

namespace kernel_features_common {

#ifdef SYCL_EXTERNAL
/**
 * @brief The external function that use T and decorated with attribute
 *
 * @tparam T The type of variable that will use inside the function
 * @tparam aspect Instance of sycl::aspect that will be used in attribute
 */
template <typename T, sycl::aspect aspect>
[[sycl::device_has(aspect)]] SYCL_EXTERNAL void
use_feature_function_external_decorated();
#endif

/**
 * @brief The function that use T
 *
 * @tparam T The type of variable that will use inside the function
 */
template <typename T>
void use_feature_function_non_decorated() {
  unsigned long long temp = 42;
  T feature(temp);
  feature += 42;
}

/**
 * @brief The function that use T and decorated with attribute
 *
 * @tparam T The type of variable that will use inside the function
 * @tparam aspect Instance of sycl::aspect that will be used in attribute
 */
template <typename T, sycl::aspect aspect>
[[sycl::device_has(aspect)]] void use_feature_function_decorated() {
  unsigned long long temp = 42;
  T feature(temp);
  feature += 42;
}

/**
 * @brief The dummy function that don't use any feature inside and decorated
 * with attribute
 *
 * @tparam aspect Instance of sycl::aspect that will be used in attribute
 */
template <sycl::aspect aspect>
[[sycl::device_has(aspect)]] void dummy_function_decorated() {
  int var = 0;
  var += 42;
}

/**
 * @brief The dummy function that don't use any feature inside
 *
 * @tparam T The type of variable that will use inside the function
 * @tparam aspect Instance of sycl::aspect that will be used in attribute
 */
void dummy_function_non_decorated() {
  int var = 0;
  var += 42;
}

/**
 * @brief Macro for generating code that will use TYPE
 */
#define USE_FEATURE(TYPE)       \
  unsigned long long temp = 42; \
  TYPE feature(temp);           \
  feature += 42;

/**
 * @brief Not decorated functor that use feature defined in FeatureTypeT
 */
template <typename FeatureTypeT>
class non_decorated_call_use_feature {
 public:
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
 * @brief Not decorated functor that invokes decorated with FeatureAspectT
 * function that use feature defined in FeatureTypeT
 */
template <typename FeatureTypeT, sycl::aspect FeatureAspectT>
class non_decorated_call_decorated_function {
 public:
  void operator()() const {
    use_feature_function_decorated<FeatureTypeT, FeatureAspectT>();
  }
  void operator()(sycl::item<1>) const {
    use_feature_function_decorated<FeatureTypeT, FeatureAspectT>();
  }
  void operator()(sycl::group<1>) const {
    use_feature_function_decorated<FeatureTypeT, FeatureAspectT>();
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
  void operator()() const {
    use_feature_function_external_decorated<FeatureTypeT, FeatureAspectT>();
  }
  void operator()(sycl::item<1>) const {
    use_feature_function_decorated<FeatureTypeT, FeatureAspectT>();
  }
  void operator()(sycl::group<1>) const {
    use_feature_function_decorated<FeatureTypeT, FeatureAspectT>();
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
  void operator()() const { dummy_function_decorated<FeatureAspectT>(); }
  void operator()(sycl::item<1>) const {
    dummy_function_decorated<FeatureAspectT>();
  }
  void operator()(sycl::group<1>) const {
    dummy_function_decorated<FeatureAspectT>();
  }
};

/**
 * @brief Decorated with FeatureAspectT functor that use feature defined in
 * FeatureTypeT
 */
template <typename FeatureTypeT, sycl::aspect FeatureAspectT>
class decorated_call_use_feature {
 public:
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
  [[sycl::device_has(KernelAspectT)]] void operator()() const {
    use_feature_function_external_decorated<FeatureTypeT, FunctionAspectT>();
  }
  [[sycl::device_has(KernelAspectT)]] void operator()(sycl::item<1>) const {
    use_feature_function_external_decorated<FeatureTypeT, FunctionAspectT>();
  }
  [[sycl::device_has(KernelAspectT)]] void operator()(sycl::group<1>) const {
    use_feature_function_external_decorated<FeatureTypeT, FunctionAspectT>();
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
  [[sycl::device_has(FeatureAspectT)]] void operator()() const {
    dummy_function_non_decorated();
  }
  [[sycl::device_has(FeatureAspectT)]] void operator()(sycl::item<1>) const {
    dummy_function_non_decorated();
  }
  [[sycl::device_has(FeatureAspectT)]] void operator()(sycl::group<1>) const {
    dummy_function_non_decorated();
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

/**
 * @brief The function helps to check if async exception was thrown from given
 * queue. The function will fail the test according to the is_exception_expected
 * parameter.
 *
 * @param queue The sycl::queue instance for exception handling
 * @param is_exception_expected The flag shows if the function will expect
 * exception or not
 */
void check_async_exception(sycl::queue &queue,
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
 * @brief The function helps to run separate lambdas in the kernel by executing
 * them in single_task, parallel_for and parallel_for_work_group. The function
 * also expects exception depending on is_exception_expected flag.
 *
 * @tparam LambdaNoArg The type of lambda for single_task invocation
 * @tparam LambdaItemArg The type of lambda for parallel_for invocation
 * @tparam LambdaGroupArg The type of lambda for parallel_for_work_group
 * invocation
 * @param is_exception_expected The flag shows if exception expected from the
 * kernel
 * @param errc_expected The error code that expected from sycl::exception
 * @param queue The sycl::queue instance for device
 * @param separate_lambda_no_arg The lambda for single_task invocation
 * @param separate_lambda_item_arg The lambda for parallel_for invocation
 * @param separate_lambda_group_arg The lambda for parallel_for_work_group
 * invocation
 */
template <typename LambdaNoArg, typename LambdaItemArg, typename LambdaGroupArg>
void run_separate_lambda(const bool is_exception_expected,
                         const sycl::errc errc_expected, sycl::queue &queue,
                         LambdaNoArg separate_lambda_no_arg,
                         LambdaItemArg separate_lambda_item_arg,
                         LambdaGroupArg separate_lambda_group_arg) {
  bool is_exception_thrown = false;
  {
    INFO("Execution of separate lambda function in single_task");
    try {
      queue
          .submit([&](sycl::handler &cgh) {
            cgh.single_task(separate_lambda_no_arg);
          })
          .wait();
      is_exception_thrown = false;
    } catch (const sycl::exception &e) {
      is_exception_thrown = true;
      if (is_exception_expected) {
        INFO("Error code check");
        CHECK(e.code() == errc_expected);
      }
    }
    CHECK(is_exception_thrown == is_exception_expected);
    check_async_exception(queue, false);
  }
  {
    INFO("Execution of separate lambda function in parallel_for");
    try {
      queue
          .submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::range{1}, separate_lambda_item_arg);
          })
          .wait();
      is_exception_thrown = false;
    } catch (const sycl::exception &e) {
      is_exception_thrown = true;
      if (is_exception_expected) {
        INFO("Error code check");
        CHECK(e.code() == errc_expected);
      }
    }
    CHECK(is_exception_thrown == is_exception_expected);
    check_async_exception(queue, false);
  }
  {
    INFO("Execution of separate lambda function in parallel_for_work_group");
    try {
      queue
          .submit([&](sycl::handler &cgh) {
            cgh.parallel_for_work_group(sycl::range{1}, sycl::range{1},
                                        separate_lambda_group_arg);
          })
          .wait();
      is_exception_thrown = false;
    } catch (const sycl::exception &e) {
      is_exception_thrown = true;
      if (is_exception_expected) {
        INFO("Error code check");
        CHECK(e.code() == errc_expected);
      }
    }
    CHECK(is_exception_thrown == is_exception_expected);
    check_async_exception(queue, false);
  }
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
  bool is_exception_thrown = false;
  {
    INFO("Execution of kernel functor in single_task");
    try {
      queue
          .submit([&](sycl::handler &cgh) {
            cgh.single_task<kernel_single_task<Functor>>(Functor{});
          })
          .wait();
      is_exception_thrown = false;
    } catch (const sycl::exception &e) {
      is_exception_thrown = true;
      if (is_exception_expected) {
        INFO("Error code check");
        CHECK(e.code() == errc_expected);
      }
    }
    CHECK(is_exception_thrown == is_exception_expected);
    check_async_exception(queue, false);
  }
  {
    INFO("Execution of kernel functor in parallel_for");
    try {
      queue
          .submit([&](sycl::handler &cgh) {
            cgh.parallel_for<kernel_parallel_for<Functor>>(sycl::range{1},
                                                           Functor{});
          })
          .wait();
      is_exception_thrown = false;
    } catch (const sycl::exception &e) {
      is_exception_thrown = true;
      if (is_exception_expected) {
        INFO("Error code check");
        CHECK(e.code() == errc_expected);
      }
    }
    CHECK(is_exception_thrown == is_exception_expected);
    check_async_exception(queue, false);
  }
  {
    INFO("Execution of kernel functor in parallel_for_work_group");
    try {
      queue
          .submit([&](sycl::handler &cgh) {
            cgh.parallel_for_work_group<kernel_parallel_for_wg<Functor>>(
                sycl::range{1}, sycl::range{1}, Functor{});
          })
          .wait();
      is_exception_thrown = false;
    } catch (const sycl::exception &e) {
      is_exception_thrown = true;
      if (is_exception_expected) {
        INFO("Error code check");
        CHECK(e.code() == errc_expected);
      }
    }
    CHECK(is_exception_thrown == is_exception_expected);
    check_async_exception(queue, false);
  }
}

#define NO_ATTRIBUTE /*no attribute*/
#define NO_KERNEL_BODY /*no kernel code*/

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
 * @param __VA_ARGS__ Body of the submission call that have to be executed on
 * the device
 */
#define RUN_SUBMISSION_CALL(IS_EXCEPTION_EXPECTED, ERRC, QUEUE, ATTRIBUTE,     \
                            ...)                                               \
  {                                                                            \
    bool is_exception_thrown = false;                                          \
                                                                               \
    {                                                                          \
      INFO("Execution of submission call in single_task");                     \
      try {                                                                    \
        QUEUE                                                                  \
            .submit([&](sycl::handler &cgh) {                                  \
              cgh.single_task([=]() ATTRIBUTE { __VA_ARGS__; });               \
            })                                                                 \
            .wait();                                                           \
        is_exception_thrown = false;                                           \
      } catch (const sycl::exception &e) {                                     \
        is_exception_thrown = true;                                            \
        if (IS_EXCEPTION_EXPECTED) {                                           \
          INFO("Error code check");                                            \
          CHECK(e.code() == ERRC);                                             \
        }                                                                      \
      }                                                                        \
      CHECK(is_exception_thrown == IS_EXCEPTION_EXPECTED);                     \
      check_async_exception(QUEUE, false);                                     \
    }                                                                          \
    {                                                                          \
      INFO("Execution of submission call in parallel_for");                    \
      try {                                                                    \
        QUEUE                                                                  \
            .submit([&](sycl::handler &cgh) {                                  \
              cgh.parallel_for(sycl::range{1},                                 \
                               [=](sycl::item<1>) ATTRIBUTE { __VA_ARGS__; }); \
            })                                                                 \
            .wait();                                                           \
        is_exception_thrown = false;                                           \
      } catch (const sycl::exception &e) {                                     \
        is_exception_thrown = true;                                            \
        if (IS_EXCEPTION_EXPECTED) {                                           \
          INFO("Error code check");                                            \
          CHECK(e.code() == ERRC);                                             \
        }                                                                      \
      }                                                                        \
      CHECK(is_exception_thrown == IS_EXCEPTION_EXPECTED);                     \
      check_async_exception(QUEUE, false);                                     \
    }                                                                          \
    {                                                                          \
      INFO("Execution of submission call in parallel_for_work_group");         \
      try {                                                                    \
        QUEUE                                                                  \
            .submit([&](sycl::handler &cgh) {                                  \
              cgh.parallel_for_work_group(sycl::range{1}, sycl::range{1},      \
                                          [=](sycl::group<1>)                  \
                                              ATTRIBUTE { __VA_ARGS__; });     \
            })                                                                 \
            .wait();                                                           \
        is_exception_thrown = false;                                           \
      } catch (const sycl::exception &e) {                                     \
        is_exception_thrown = true;                                            \
        if (IS_EXCEPTION_EXPECTED) {                                           \
          INFO("Error code check");                                            \
          CHECK(e.code() == ERRC);                                             \
        }                                                                      \
      }                                                                        \
      CHECK(is_exception_thrown == IS_EXCEPTION_EXPECTED);                     \
      check_async_exception(QUEUE, false);                                     \
    }                                                                          \
  }
};  // namespace kernel_features_common

#endif  // SYCL_CTS_TEST_KERNEL_FEATURES_COMMON_H

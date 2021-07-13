/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides common methods for group async_work_group_copy tests
//
*******************************************************************************/

#ifndef SYCL_1_2_1_TESTS_GROUP_ASYNC_WORK_GROUP_COPY_COMMON_H
#define SYCL_1_2_1_TESTS_GROUP_ASYNC_WORK_GROUP_COPY_COMMON_H

#include "../common/common.h"
#include "../common/async_work_group_copy.h"
#include "../common/invoke.h"

// Enforce ODR
namespace group_async_work_group_copy {

template<typename T, int dim>
class kernel_type;

/**
 * @brief Makes common test logic call with appropriate invocation functor type
 *        provided
 * @tparam dim Dimension to use for group instance
 * @tparam T Type to use for group::async_work_group_copy() call
 */
template<int dim, typename T>
struct check_dim {
  /**
   * @param queue SYCL queue to use for test
   * @param log Logger to use for test
   * @param typeName The string naming the type of data for logs
   */
  void operator()(sycl::queue &queue, sycl_cts::util::logger &log,
                  const std::string& typeName) {
    using kernelT = kernel_type<T, dim>;
    using kernelInvokeT = invoke_group<dim, kernelT>;
    static const std::string instanceName = "group";

    test_async_wg_copy<kernelInvokeT, T>(queue, log, instanceName, typeName);
  }
};

/**
 * @brief Syntax sugar wrapping the call for all dimensions, with or without
 *        queue provided
 * @tparam T Type to use for group::async_work_group_copy() call
 */
template<typename T>
struct check_type {
  /**
   * @tparam argsT Deduced parameter pack for types of arguments
   * @param args Arguments provided; usage of the queue as the first argument
   *             defines actual overload called
   */
  template<typename ... argsT>
  void operator()(argsT&& ... args) {
    check_all_dims<check_dim, T>(std::forward<argsT>(args)...);
  }
};

} // namespace group_async_work_group_copy

#endif // SYCL_1_2_1_TESTS_GROUP_ASYNC_WORK_GROUP_COPY_COMMON_H

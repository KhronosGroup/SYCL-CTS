/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Common checks for specialization constants with SYCL_EXTERNAL function
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_SPEC_CONST_EXTERNAL_H
#define __SYCLCTS_TESTS_SPEC_CONST_EXTERNAL_H

#include "../common/common.h"
#include "../common/type_list.h"
#include "spec_constants_common.h"

using namespace get_spec_const;

template <typename T, int case_num>
inline constexpr sycl::specialization_id<T> spec_const_external(
    user_def_types::get_init_value<T>(default_val));

#define FUNC_DECLARE(TYPE)                                               \
  SYCL_EXTERNAL bool check_kernel_handler_by_reference_external_handler( \
      sycl::kernel_handler &h, TYPE);                                    \
  SYCL_EXTERNAL bool check_kernel_handler_by_value_external_handler(     \
      sycl::kernel_handler h, TYPE);                                     \
  SYCL_EXTERNAL bool check_kernel_handler_by_reference_external_bundle(  \
      sycl::kernel_handler &h, TYPE);                                    \
  SYCL_EXTERNAL bool check_kernel_handler_by_value_external_bundle(      \
      sycl::kernel_handler h, TYPE);

#ifdef TEST_CORE
#if !SYCL_CTS_ENABLE_FULL_CONFORMANCE
CORE_TYPES(FUNC_DECLARE)
#else
CORE_TYPES_PARAM(SYCL_VECTORS_MARRAYS, FUNC_DECLARE)
#endif
FUNC_DECLARE(user_def_types::no_cnstr)
FUNC_DECLARE(user_def_types::def_cnstr)
FUNC_DECLARE(user_def_types::no_def_cnstr)
#endif  // TEST_CORE

#ifdef TEST_FP64
#if !SYCL_CTS_ENABLE_FULL_CONFORMANCE
FUNC_DECLARE(double)
#else
SYCL_VECTORS_MARRAYS(double, FUNC_DECLARE)
#endif
#endif  // TEST_FP64

#ifdef TEST_FP16
#if !SYCL_CTS_ENABLE_FULL_CONFORMANCE
FUNC_DECLARE(sycl::half)
#else
SYCL_VECTORS_MARRAYS(sycl::half, FUNC_DECLARE)
#endif
#endif  // TEST_FP16

namespace specialization_constants_external {

template <typename T, int num_case>
class kernel;

using namespace sycl_cts;

template <typename T>
class check_specialization_constants_external {
 public:
  void operator()(util::logger &log, const std::string &type_name) {
    auto queue = util::get_cts_object::queue();
    sycl::range<1> range(1);

    // case 1: Pass kernel handler object by reference to external function via
    // handler
    bool passed = false;
    {
      T ref{user_def_types::get_init_value<T>(5)};
      const int case_num =
          static_cast<int>(test_cases_external::by_reference_via_handler);
      sycl::buffer<bool, 1> result_buffer(&passed, range);
      queue.submit([&](sycl::handler &cgh) {
        auto res_acc =
            result_buffer.template get_access<sycl::access_mode::write>(cgh);
        cgh.set_specialization_constant<spec_const_external<T, case_num>>(ref);
        cgh.single_task<kernel<T, case_num>>([=](sycl::kernel_handler h) {
          res_acc[0] =
              check_kernel_handler_by_reference_external_handler(h, ref);
        });
      });
    }
    if (!passed) {
      FAIL(log,
           "case 1: Pass kernel handler object by reference to external "
           "function via handler failed for " +
               type_name_string<T>::get(type_name));
    }

    // case 2: Pass kernel handler object by value to external function via
    // handler
    passed = false;
    {
      T ref{user_def_types::get_init_value<T>(10)};
      const int case_num =
          static_cast<int>(test_cases_external::by_value_via_handler);
      sycl::buffer<bool, 1> result_buffer(&passed, range);
      queue.submit([&](sycl::handler &cgh) {
        auto res_acc =
            result_buffer.template get_access<sycl::access_mode::write>(cgh);
        cgh.set_specialization_constant<spec_const_external<T, case_num>>(ref);
        cgh.single_task<kernel<T, case_num>>([=](sycl::kernel_handler h) {
          res_acc[0] = check_kernel_handler_by_value_external_handler(h, ref);
        });
      });
    }
    if (!passed) {
      FAIL(log,
           "case 2: Pass kernel handler object by value to external function "
           "via handler failed for " +
               type_name_string<T>::get(type_name));
    }

    if (!queue.get_device().has(sycl::aspect::online_compiler)) {
      WARN("Device does not support online compilation of device code");
    } else {
      const auto context = queue.get_context();

      // case 3: Pass kernel handler object by reference to external function
      // via kernel_bundle
      passed = false;
      {
        T ref{user_def_types::get_init_value<T>(15)};
        const int case_num =
            static_cast<int>(test_cases_external::by_reference_via_bundle);
        sycl::buffer<bool, 1> result_buffer(&passed, range);

        sycl::kernel_id kernelID = sycl::get_kernel_id<kernel<T, case_num>>();
        auto inputBundle = sycl::get_kernel_bundle<sycl::bundle_state::input>(
            context, {kernelID});
        if (!inputBundle.has_kernel(kernelID)) {
          // It's implementation-defined if a kernel is available in a bundle
          // with bundle_state::input. So if such bundle misses some kernels it
          // shouldn't trigger a test failure.
          WARN("Input bundle misses kernel in question");
          passed = true;
        } else {
          inputBundle.template set_specialization_constant<
              spec_const_external<T, case_num>>(ref);
          auto exeBundle = sycl::build(inputBundle);

          queue.submit([&](sycl::handler &cgh) {
            auto res_acc =
                result_buffer.template get_access<sycl::access_mode::write>(
                    cgh);
            cgh.use_kernel_bundle(exeBundle);
            cgh.single_task<kernel<T, case_num>>([=](sycl::kernel_handler h) {
              res_acc[0] =
                  check_kernel_handler_by_reference_external_bundle(h, ref);
            });
          });
        }
      }
      if (!passed) {
        FAIL(log,
             "case 3: Pass kernel handler object by reference to external "
             "function via  kernel_bundle failed for " +
                 type_name_string<T>::get(type_name));
      }

      // case 4: Pass kernel handler object by value to external function via
      // kernel_bundle
      passed = false;
      {
        T ref{user_def_types::get_init_value<T>(20)};
        const int case_num =
            static_cast<int>(test_cases_external::by_value_via_bundle);
        sycl::buffer<bool, 1> result_buffer(&passed, range);

        sycl::kernel_id kernelID = sycl::get_kernel_id<kernel<T, case_num>>();
        auto inputBundle = sycl::get_kernel_bundle<sycl::bundle_state::input>(
            context, {kernelID});
        if (!inputBundle.has_kernel(kernelID)) {
          // It's implementation-defined if a kernel is available in a bundle
          // with bundle_state::input.
          WARN("Input bundle misses kernel in question");
          passed = true;
        } else {
          inputBundle.template set_specialization_constant<
              spec_const_external<T, case_num>>(ref);
          auto exeBundle = sycl::build(inputBundle);

          queue.submit([&](sycl::handler &cgh) {
            auto res_acc =
                result_buffer.template get_access<sycl::access_mode::write>(
                    cgh);
            cgh.use_kernel_bundle(exeBundle);
            cgh.single_task<kernel<T, case_num>>([=](sycl::kernel_handler h) {
              res_acc[0] =
                  check_kernel_handler_by_value_external_bundle(h, ref);
            });
          });
        }
      }
      if (!passed) {
        FAIL(log,
             "case 4: Pass kernel handler object by value to external function "
             "via kernel_bundle failed for " +
                 type_name_string<T>::get(type_name));
      }
    }
  }
};
} /* namespace specialization_constants_external */
#endif  // __SYCLCTS_TESTS_SPEC_CONST_EXTERNAL_H

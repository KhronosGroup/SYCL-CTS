/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Common code for expected exceptions throwing by specialization constants.
//  In this tests we check that exception with code sycl::errc::invalid is thrown
//  any other exception causes test to fail and gets logged.
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_SPEC_CONST_THROW_EXCEPT_COMMON_H
#define __SYCLCTS_TESTS_SPEC_CONST_THROW_EXCEPT_COMMON_H

#include "specialization_constants_common.h"

template <typename T>
class dummy_specialization_constants_exceptions {};
template <typename T>
using spec_const_exception_dummy_functor =
    ::dummy_functor<dummy_specialization_constants_exceptions<T>>;

template <typename T>
class check_spec_constant_exception_throw_for_type {
 public:
  void operator()(sycl_cts::util::logger &log, const std::string &type_name) {
    using namespace get_spec_const;
    const std::string err_message_prefix{
        "unexpected SYCL exception was thrown in case "};

    // case 1: Try to get specialization constant via handler that is bound to a
    // kernel_bundle
    {
      bool exception_was_thrown = false;
      T res { user_def_types::get_init_value_helper<T>(0) };
      const int case_num = 1;
      auto queue = sycl_cts::util::get_cts_object::queue();

      queue.submit([&](sycl::handler &cgh) {
        auto context = queue.get_context();
        auto k_bundle =
            sycl::get_kernel_bundle<sycl::bundle_state::executable>(context);
        cgh.use_kernel_bundle(k_bundle);
        // We expect that exception with sycl::errc::invalid will be thrown
        try {
          res = cgh.get_specialization_constant<spec_const<T, case_num>>();
        } catch (const sycl::exception &e) {
          if (e.code() != sycl::errc::invalid) {
            const auto errorMsg =
                "unexpected SYCL exception was thrown in case " +
                std::to_string(case_num) + " for " +
                type_name_string<T>::get(type_name);
            FAIL(log, errorMsg);
            throw;
          } else {
            exception_was_thrown = true;
          }
        }
        cgh.single_task(spec_const_exception_dummy_functor<T>{});
      });
      CHECK_VALUE_SCALAR(log, exception_was_thrown, true);
    }

    // case 2: Try to set specialization constant via handler that is bound to a
    // kernel_bundle
    {
      bool exception_was_thrown = false;
      T sc_val { user_def_types::get_init_value_helper<T>(0) };
      const int case_num = 2;
      auto queue = sycl_cts::util::get_cts_object::queue();

      queue.submit([&](sycl::handler &cgh) {
        auto context = queue.get_context();
        auto k_bundle =
            sycl::get_kernel_bundle<sycl::bundle_state::executable>(context);
        cgh.use_kernel_bundle(k_bundle);
        // We expect that exception with sycl::errc::invalid will be thrown
        try {
          cgh.set_specialization_constant<spec_const<T, case_num>>(sc_val);
        } catch (const sycl::exception &e) {
          if (static_cast<sycl::errc>(e.code().value()) !=
              sycl::errc::invalid) {
            const auto errorMsg =
                "unexpected SYCL exception was thrown in case " +
                std::to_string(case_num) + " for " +
                type_name_string<T>::get(type_name);
            FAIL(log, errorMsg);
            throw;
          } else {
            exception_was_thrown = true;
          }
        }
        cgh.single_task(spec_const_exception_dummy_functor<T>{});
      });
      CHECK_VALUE_SCALAR(log, exception_was_thrown, true);
    }
  }
};

#endif  // __SYCLCTS_TESTS_SPEC_CONST_THROW_EXCEPT_COMMON_H

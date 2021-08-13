/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Common expected exceptions catching for specialization constants. In this
//  tests we add error message that check exception's type, but the exception
//  logging in higher level.
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_SPEC_CONST_EXCEPT_CATCH_COMMON_H
#define __SYCLCTS_TESTS_SPEC_CONST_EXCEPT_CATCH_COMMON_H

#include "specialization_constants_common.h"

template <typename T>
class dummy_specialization_constants_exceptions {};
template <typename T>
using spec_const_except_dummy_functor =
    ::dummy_functor<dummy_specialization_constants_exceptions<T>>;

template <typename T>
class check_spec_constant_except_catch_for_type {
 public:
  void operator()(sycl_cts::util::logger &log, const std::string &type_name) {
    using namespace get_spec_const;
    const std::string err_message_prefix{
        "unexpected SYCL exception was caught in case "};

    // case 1: Try to get specialization constant via handler that bound to a
    // kernel_bundle
    {
      T res = T(value_helper<T>(0));
      const int case_num = 1;
      auto queue = sycl_cts::util::get_cts_object::queue();

      queue.submit([&](sycl::handler &cgh) {
        auto context = queue.get_context();
        auto k_bundle =
            sycl::get_kernel_bundle<sycl::bundle_state::executable>(context);
        cgh.use_kernel_bundle(k_bundle);
        try {
          res = cgh.get_specialization_constant<spec_const<T, case_num>>();
        } catch (const sycl::exception &e) {
          if (e.code() != sycl::errc::invalid) {
            const auto errorMsg =
                "unexpected SYCL exception was caught in case 1 for " +
                type_name_string<T>::get(type_name);
            FAIL(log, errorMsg);
            throw;
          }
        }
        cgh.single_task(spec_const_except_dummy_functor<T>{});
      });
    }

    // case 2: Try to set specialization constant via handler that bound to a
    // kernel_bundle
    {
      const int case_num = 2;
      T sc_val = T(value_helper<T>(0));
      auto queue = sycl_cts::util::get_cts_object::queue();

      queue.submit([&](sycl::handler &cgh) {
        auto context = queue.get_context();
        auto k_bundle =
            sycl::get_kernel_bundle<sycl::bundle_state::executable>(context);
        cgh.use_kernel_bundle(k_bundle);
        try {
          cgh.set_specialization_constant<spec_const<T, case_num>>(sc_val);
        } catch (const sycl::exception &e) {
          if (static_cast<sycl::errc>(e.code().value()) != sycl::errc::invalid) {
            const auto errorMsg =
                "unexpected SYCL exception was caught in case 2 for " +
                type_name_string<T>::get(type_name);
            FAIL(log, errorMsg);
            throw;
          }
        }
        cgh.single_task(spec_const_except_dummy_functor<T>{});
      });
    }
  }
};

#endif  // __SYCLCTS_TESTS_SPEC_CONST_EXCEPT_CATCH_COMMON_H

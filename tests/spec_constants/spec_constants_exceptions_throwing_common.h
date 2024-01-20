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

//  Common code for expected exceptions throwing by specialization constants.
//  In this tests we check that exception with code sycl::errc::invalid is
//  thrown any other exception causes test to fail and gets logged.

#ifndef __SYCLCTS_TESTS_SPEC_CONST_THROW_EXCEPT_COMMON_H
#define __SYCLCTS_TESTS_SPEC_CONST_THROW_EXCEPT_COMMON_H

#include "spec_constants_common.h"

template <typename T>
class dummy_specialization_constants_exceptions {};
template <typename T>
using spec_const_exception_dummy_functor =
    ::dummy_functor<dummy_specialization_constants_exceptions<T>>;

template <typename T>
class kernel_spec_constant_exception_get;

template <typename T>
class kernel_spec_constant_exception_set;

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
      T res{user_def_types::get_init_value<T>(0)};
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
        cgh.single_task<kernel_spec_constant_exception_get<T>>(
            spec_const_exception_dummy_functor<T>{});
      });
      CHECK_VALUE_SCALAR(log, exception_was_thrown, true);
    }

    // case 2: Try to set specialization constant via handler that is bound to a
    // kernel_bundle
    {
      bool exception_was_thrown = false;
      T sc_val{user_def_types::get_init_value<T>(0)};
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
        cgh.single_task<kernel_spec_constant_exception_set<T>>(
            spec_const_exception_dummy_functor<T>{});
      });
      CHECK_VALUE_SCALAR(log, exception_was_thrown, true);
    }
  }
};

#endif  // __SYCLCTS_TESTS_SPEC_CONST_THROW_EXCEPT_COMMON_H

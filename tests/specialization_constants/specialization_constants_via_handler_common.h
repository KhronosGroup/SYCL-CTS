/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
// Common checks for specialization constants usage via handler
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_SPEC_CONST_HANDLER_COMMON_H
#define __SYCLCTS_TESTS_SPEC_CONST_HANDLER_COMMON_H

#include "../../util/math_helper.h"
#include "../common/common.h"
#include "specialization_constants_common.h"

namespace specialization_constants_via_handler_common {
using namespace sycl_cts;
using namespace get_spec_const;

inline constexpr int val_A = 5;

template <typename T, int case_num>
constexpr sycl::specialization_id<T> sc_multiple(value_helper<T>(case_num));

template <typename T, int case_num> class kernel;

template <typename T, int case_num>
bool check_kernel_handler_by_reference(sycl::kernel_handler &h) {
  T ref = T(value_helper<T>(0));
  init_values(ref, val_A);
  return check_equal_values(
      ref, h.get_specialization_constant<spec_const<T, case_num>>());
}

template <typename T, int case_num>
bool check_kernel_handler_by_value(sycl::kernel_handler h) {
  T ref = T(value_helper<T>(0));
  init_values(ref, val_A);
  return check_equal_values(
      ref, h.get_specialization_constant<spec_const<T, case_num>>());
}

template <typename T> class check_spec_constant_with_handler_for_type {
public:
  void operator()(util::logger &log, const std::string &type_name) {
    auto queue = util::get_cts_object::queue();
    sycl::range<1> range(1);
    T result = T(value_helper<T>(0));
    T ref = T(value_helper<T>(0));
    T ref_other = T(value_helper<T>(0));
    int val_B = 10;
    int val_C = 30;
    init_values(ref, val_A);
    init_values(ref_other, val_B);
    // case 1: Set the value in the handler and read it from the same handler.
    {
      const int case_num = 1;
      result = T(value_helper<T>(0));
      queue.submit([&](sycl::handler &cgh) {
        cgh.set_specialization_constant<spec_const<T, case_num>>(ref);
        result = cgh.get_specialization_constant<spec_const<T, case_num>>();
      });
    }
    if (!check_equal_values(ref, result))
      FAIL(log, "case 1 for " + type_name_string<T>::get(type_name));

    // case 2: Set the value in the handler twice and read it from the same
    // handler.
    {
      const int case_num = 2;
      result = T(value_helper<T>(0));
      queue.submit([&](sycl::handler &cgh) {
        cgh.set_specialization_constant<spec_const<T, case_num>>(ref);
        cgh.set_specialization_constant<spec_const<T, case_num>>(ref_other);
        result = cgh.get_specialization_constant<spec_const<T, case_num>>();
      });
    }
    if (!check_equal_values(ref_other, result))
      FAIL(log, "case 2 for " + type_name_string<T>::get(type_name));

    // case 3: Set the value in the handler, launch a kernel, and read the value
    // from the kernel.
    {
      const int case_num = 3;
      result = T(value_helper<T>(0));
      sycl::buffer<T, 1> result_buffer(&result, range);
      queue.submit([&](sycl::handler &cgh) {
        auto res_acc =
            result_buffer.template get_access<sycl::access_mode::write>(cgh);
        cgh.set_specialization_constant<spec_const<T, case_num>>(ref);
        cgh.single_task<kernel<T, case_num>>([=](sycl::kernel_handler h) {
          res_acc[0] = h.get_specialization_constant<spec_const<T, case_num>>();
        });
      });
    }
    if (!check_equal_values(ref, result))
      FAIL(log, "case 3 for " + type_name_string<T>::get(type_name));

    // case 4: Set the value in the handler twice, launch a kernel, and read the
    // value from the kernel.
    {
      const int case_num = 4;
      result = T(value_helper<T>(0));
      sycl::buffer<T, 1> result_buffer(&result, range);
      queue.submit([&](sycl::handler &cgh) {
        auto res_acc =
            result_buffer.template get_access<sycl::access_mode::write>(cgh);
        cgh.set_specialization_constant<spec_const<T, case_num>>(ref);
        cgh.set_specialization_constant<spec_const<T, case_num>>(ref_other);
        cgh.single_task<kernel<T, case_num>>([=](sycl::kernel_handler h) {
          res_acc[0] = h.get_specialization_constant<spec_const<T, case_num>>();
        });
      });
    }
    if (!check_equal_values(ref_other, result))
      FAIL(log, "case 4 for " + type_name_string<T>::get(type_name));

    // case 5: Set the value in the handler, launch a kernel,
    // and read the value from the kernel twice.
    size_t size = 2;
    // to not initialize for struct with no default constructor
    T *result_vec_same = (T *)malloc(size * sizeof(T));
    {
      const int case_num = 5;
      sycl::buffer<T, 1> result_buffer(result_vec_same, sycl::range<1>(size));
      queue.submit([&](sycl::handler &cgh) {
        auto res_acc =
            result_buffer.template get_access<sycl::access_mode::write>(cgh);
        cgh.set_specialization_constant<spec_const<T, case_num>>(ref);
        cgh.single_task<kernel<T, case_num>>([=](sycl::kernel_handler h) {
          res_acc[0] = h.get_specialization_constant<spec_const<T, case_num>>();
          res_acc[1] = h.get_specialization_constant<spec_const<T, case_num>>();
        });
      });
    }
    if (!check_equal_values(ref, result_vec_same[0]) ||
        !check_equal_values(ref, result_vec_same[1]))
      FAIL(log, "case 5 for " + type_name_string<T>::get(type_name));

    // case 6: Do not set the value of the spec constant, and read it from the
    // handler. Expecting default value.
    {
      const int case_num = 6;
      result = T(value_helper<T>(0));
      queue.submit([&](sycl::handler &cgh) {
        result = cgh.get_specialization_constant<spec_const<T, case_num>>();
      });
    }
    if (!check_equal_values(T(value_helper<T>(default_val)), result))
      FAIL(log, "case 6 for " + type_name_string<T>::get(type_name));

    // case 7: Do not set the value of the spec constant, launch a kernel, and
    // read the value from the kernel. Expecting default value.
    {
      const int case_num = 7;
      result = T(value_helper<T>(0));
      sycl::buffer<T, 1> result_buffer(&result, range);
      queue.submit([&](sycl::handler &cgh) {
        auto res_acc =
            result_buffer.template get_access<sycl::access_mode::write>(cgh);
        cgh.single_task<kernel<T, case_num>>([=](sycl::kernel_handler h) {
          res_acc[0] = h.get_specialization_constant<spec_const<T, case_num>>();
        });
      });
    }
    if (!check_equal_values(T(value_helper<T>(default_val)), result))
      FAIL(log, "case 7 for " + type_name_string<T>::get(type_name));

    // case 8: Pass kernel handler object by reference to another function
    bool func_result = false;
    {
      const int case_num = 8;
      sycl::buffer<bool, 1> result_buffer(&func_result, range);
      queue.submit([&](sycl::handler &cgh) {
        auto res_acc =
            result_buffer.template get_access<sycl::access_mode::write>(cgh);
        cgh.set_specialization_constant<spec_const<T, case_num>>(ref);
        cgh.single_task<kernel<T, case_num>>([=](sycl::kernel_handler h) {
          res_acc[0] = check_kernel_handler_by_reference<T, case_num>(h);
        });
      });
    }
    if (!func_result)
      FAIL(log, "case 8 for " + type_name_string<T>::get(type_name));

    // case 9: Pass kernel handler object by value to another function
    func_result = false;
    {
      const int case_num = 9;
      sycl::buffer<bool, 1> result_buffer(&func_result, range);
      queue.submit([&](sycl::handler &cgh) {
        auto res_acc =
            result_buffer.template get_access<sycl::access_mode::write>(cgh);
        cgh.set_specialization_constant<spec_const<T, case_num>>(ref);
        cgh.single_task<kernel<T, case_num>>([=](sycl::kernel_handler h) {
          res_acc[0] = check_kernel_handler_by_value<T, case_num>(h);
        });
      });
    }
    if (!func_result)
      FAIL(log, "case 9 for " + type_name_string<T>::get(type_name));
    free(result_vec_same);
  }
};
} /* namespace specialization_constants_via_handler_common */
#endif // __SYCLCTS_TESTS_SPEC_CONST_HANDLER_COMMON_H

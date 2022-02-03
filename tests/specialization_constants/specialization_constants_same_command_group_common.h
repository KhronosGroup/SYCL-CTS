/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
// Common checks for specialization constants usage via handler
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_SPEC_CONST_SAME_COMMAND_GROUP_COMMON_H
#define __SYCLCTS_TESTS_SPEC_CONST_SAME_COMMAND_GROUP_COMMON_H

#include "../common/common.h"
#include "specialization_constants_common.h"

namespace specialization_constants_same_command_group_common {
using namespace sycl_cts;
using namespace get_spec_const;

template <typename T, int num_case>
class kernel;

template <typename T, int num_case>
class command_group_object {
  T *value;  // to not initialize for struct with no default constructor
  sycl::buffer<T, 1> *result_buffer;

 public:
  bool set_const;
  void set_value(T *value_) {
    value = value_;
    set_const = true;
  }
  void set_buffer(sycl::buffer<T, 1> *buffer) { result_buffer = buffer; }
  void operator()(sycl::handler &cgh) {
    if (set_const)
      cgh.set_specialization_constant<spec_const<T, num_case>>(*value);
    auto res_acc =
        result_buffer->template get_access<sycl::access_mode::write>(cgh);
    cgh.single_task<kernel<T, num_case>>([=](sycl::kernel_handler h) {
      res_acc[0] = h.get_specialization_constant<spec_const<T, num_case>>();
    });
  }
};

template <typename T>
class check_specialization_constants_same_command_group {
 public:
  void operator()(util::logger &log, const std::string &type_name) {
    T ref_A{user_def_types::get_init_value_helper<T>(5)};
    T ref_B{user_def_types::get_init_value_helper<T>(10)};
    auto queue = util::get_cts_object::queue();
    sycl::range<1> range(1);
    {
      T result1{user_def_types::get_init_value_helper<T>(0)};
      T result2{user_def_types::get_init_value_helper<T>(0)};
      {
        command_group_object<T, 1> cmo;
        sycl::buffer<T> result_buffer1(&result1, range);
        sycl::buffer<T> result_buffer2(&result2, range);

        cmo.set_value(&ref_A);
        cmo.set_buffer(&result_buffer1);
        queue.submit(cmo);

        cmo.set_value(&ref_B);
        cmo.set_buffer(&result_buffer2);
        queue.submit(cmo);
      }
      if (!check_equal_values(ref_A, result1))
        FAIL(log, "case 1 failed for value A for " + type_name);
      if (!check_equal_values(ref_B, result2))
        FAIL(log, "case 1 failed for value B for " + type_name);
    }

    {
      T result1{user_def_types::get_init_value_helper<T>(0)};
      T result2{user_def_types::get_init_value_helper<T>(0)};
      {
        command_group_object<T, 2> cmo;
        sycl::buffer<T> result_buffer1(&result1, range);
        sycl::buffer<T> result_buffer2(&result2, range);

        cmo.set_value(&ref_A);
        cmo.set_buffer(&result_buffer1);
        queue.submit(cmo);

        cmo.set_const = false;
        cmo.set_buffer(&result_buffer2);
        queue.submit(cmo);
      }
      if (!check_equal_values(ref_A, result1))
        FAIL(log, "case 2 failed for value A for " + type_name);
      if (!check_equal_values(
              T(user_def_types::get_init_value_helper<T>(default_val)),
              result2))
        FAIL(log, "case 2 failed for default value for " + type_name);
    }
  }
};
} /* namespace specialization_constants_same_command_group_common */
#endif  // __SYCLCTS_TESTS_SPEC_CONST_SAME_COMMAND_GROUP_COMMON_H

/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Common checks for multiple specialization constants via kernel_bundle
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_SPEC_CONST_MULTIPLE_H
#define __SYCLCTS_TESTS_SPEC_CONST_MULTIPLE_H

#include "../common/common.h"
#include "../common/once_per_unit.h"
#include "specialization_constants_common.h"

template <typename T, int def_val>
constexpr sycl::specialization_id<T> sc_multiple(
    get_spec_const::get_init_value_helper<T>(def_val));

namespace specialization_constants_multiple {
using namespace sycl_cts;
using namespace get_spec_const;

template <typename T, typename via_kb>
class sc_kernel_multiple;

constexpr int size = 6;
constexpr std::array<int, size> def_values{11, 22, 33, 44, 55, 66};

template <typename T, typename ResAccType>
void get_values_from_handler(ResAccType &res_acc, sycl::kernel_handler &h) {
  res_acc[0] = h.get_specialization_constant<sc_multiple<T, def_values[0]>>();
  res_acc[1] = h.get_specialization_constant<sc_multiple<T, def_values[1]>>();
  res_acc[2] = h.get_specialization_constant<sc_multiple<T, def_values[2]>>();
  res_acc[3] = h.get_specialization_constant<sc_multiple<T, def_values[3]>>();
  res_acc[4] = h.get_specialization_constant<sc_multiple<T, def_values[4]>>();
  res_acc[5] = h.get_specialization_constant<sc_multiple<T, def_values[5]>>();
}

template <typename T, typename via_kb>
class check_specialization_constants_multiple_for_type {
 public:
  void operator()(util::logger &log, const std::string &type_name) {
    // Set several values via kernel_bundle, launch a kernel, and read
    // these values and some default values from the kernel.
    auto queue = util::get_cts_object::queue();
    const sycl::context ctx = queue.get_context();
    const sycl::device dev = queue.get_device();

    if constexpr (via_kb::value) {
      if (!dev.has(sycl::aspect::online_compiler)) {
        once_per_unit::log(log, "Device does not support online compilation");
        return;
      }
    }
    int val_A = 5;
    int val_B = 10;
    int val_C = 30;
    T ref1 = T(get_init_value_helper<T>(0));
    T ref2 = T(get_init_value_helper<T>(0));
    T ref3 = T(get_init_value_helper<T>(0));
    fill_init_values(ref1, val_A);
    fill_init_values(ref2, val_B);
    fill_init_values(ref3, val_C);
    // to not initialize for struct with no default constructor
    T *result_vec = (T *)malloc(size * sizeof(T));
    {
      sycl::buffer<T, 1> result_buffer(result_vec, sycl::range<1>(size));
      queue.submit([&](sycl::handler &cgh) {
        auto res_acc =
            result_buffer.template get_access<sycl::access_mode::write>(cgh);
        // Via kernel_bundle
        if constexpr (via_kb::value) {
          auto kernelId = sycl::get_kernel_id<sc_kernel_multiple<T, via_kb>>();
          sycl::kernel_bundle k_bundle =
              sycl::get_kernel_bundle<sycl::bundle_state::input>(ctx, {dev},
                                                                 {kernelId});
          if (!k_bundle.has_kernel(kernelId)) {
            log.note(
                "kernel_bundle doesn't contain target kernel;"
                "multiple spec const for " +
                type_name_string<T>::get(type_name) + "(skipped)");
            return;
          }

          k_bundle.template set_specialization_constant<
              sc_multiple<T, def_values[0]>>(ref1);
          k_bundle.template set_specialization_constant<
              sc_multiple<T, def_values[1]>>(ref2);
          k_bundle.template set_specialization_constant<
              sc_multiple<T, def_values[2]>>(ref3);

          auto exec_bundle = sycl::build(k_bundle);
          cgh.use_kernel_bundle(exec_bundle);

          cgh.single_task<sc_kernel_multiple<T, via_kb>>(
              [=](sycl::kernel_handler h) {
                get_values_from_handler<T>(res_acc, h);
              });
        } else {
          // No kernel_bundle
          cgh.set_specialization_constant<sc_multiple<T, def_values[0]>>(ref1);
          cgh.set_specialization_constant<sc_multiple<T, def_values[1]>>(ref2);
          cgh.set_specialization_constant<sc_multiple<T, def_values[2]>>(ref3);

          cgh.single_task<sc_kernel_multiple<T, via_kb>>(
              [=](sycl::kernel_handler h) {
                get_values_from_handler<T>(res_acc, h);
              });
        }
      });
    }
    if (!check_equal_values(ref1, result_vec[0]) ||
        !check_equal_values(ref2, result_vec[1]) ||
        !check_equal_values(ref3, result_vec[2]) ||
        !check_equal_values(T(get_init_value_helper<T>(def_values[3])),
                            result_vec[3]) ||
        !check_equal_values(T(get_init_value_helper<T>(def_values[4])),
                            result_vec[4]) ||
        !check_equal_values(T(get_init_value_helper<T>(def_values[5])),
                            result_vec[5]))
      FAIL(log,
           "multiple spec const for " + type_name_string<T>::get(type_name));

    free(result_vec);
  }
};

template <typename via_kb>
static void sc_run_test_core(util::logger &log) {
  using namespace specialization_constants_multiple;
  try {
#ifndef SYCL_CTS_FULL_CONFORMANCE
    for_all_types<check_specialization_constants_multiple_for_type, via_kb>(
        get_spec_const::testing_types::types, log);
#else
    for_all_types_vectors_marray<
        check_specialization_constants_multiple_for_type, via_kb>(
        get_spec_const::testing_types::types, log);
#endif
    for_all_types<check_specialization_constants_multiple_for_type, via_kb>(
        get_spec_const::testing_types::composite_types, log);

  } catch (const sycl::exception &e) {
    log_exception(log, e);
    std::string errorMsg =
        "a SYCL exception was caught: " + std::string(e.what());
    FAIL(log, errorMsg);
  } catch (const std::exception &e) {
    std::string errorMsg = "an exception was caught: " + std::string(e.what());
    FAIL(log, errorMsg);
  }
}

template <typename via_kb>
static void sc_run_test_fp16(util::logger &log) {
  using namespace specialization_constants_multiple;
  try {
    auto queue = util::get_cts_object::queue();
    if (!queue.get_device().has(sycl::aspect::fp16)) {
      log.note(
          "Device does not support half precision floating point "
          "operations");
      return;
    }
#ifndef SYCL_CTS_FULL_CONFORMANCE
    check_specialization_constants_multiple_for_type<sycl::half, via_kb>
        fp16_test{};
    fp16_test(log, "sycl::half");
#else
    for_type_vectors_marray<check_specialization_constants_multiple_for_type,
                            sycl::half, via_kb>(log, "sycl::half");
#endif

  } catch (const sycl::exception &e) {
    log_exception(log, e);
    std::string errorMsg =
        "a SYCL exception was caught: " + std::string(e.what());
    FAIL(log, errorMsg);
  } catch (const std::exception &e) {
    std::string errorMsg = "an exception was caught: " + std::string(e.what());
    FAIL(log, errorMsg);
  }
}

template <typename via_kb>
static void sc_run_test_fp64(util::logger &log) {
  using namespace specialization_constants_multiple;
  try {
    auto queue = util::get_cts_object::queue();
    if (!queue.get_device().has(sycl::aspect::fp64)) {
      log.note(
          "Device does not support double precision floating point "
          "operations");
      return;
    }
#ifndef SYCL_CTS_FULL_CONFORMANCE
    check_specialization_constants_multiple_for_type<double, via_kb>
        fp64_test{};
    fp64_test(log, "double");
#else
    for_type_vectors_marray<check_specialization_constants_multiple_for_type,
                            double, via_kb>(log, "double");
#endif

  } catch (const sycl::exception &e) {
    log_exception(log, e);
    std::string errorMsg =
        "a SYCL exception was caught: " + std::string(e.what());
    FAIL(log, errorMsg);
  } catch (const std::exception &e) {
    std::string errorMsg = "an exception was caught: " + std::string(e.what());
    FAIL(log, errorMsg);
  }
}

}  // namespace specialization_constants_multiple

#endif  // __SYCLCTS_TESTS_SPEC_CONST_MULTIPLE_H

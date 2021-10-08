/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Common checks for specialization constants same name stress test
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_SPEC_CONST_SAME_NAME_STRESS_H
#define __SYCLCTS_TESTS_SPEC_CONST_SAME_NAME_STRESS_H

#include "../common/common.h"
#include "../common/once_per_unit.h"
#include "../../util/allocation.h"
#include "specialization_constants_common.h"
#include "specialization_constants_same_name_stress_helper.h"

namespace specialization_constants_same_name_stress {
using namespace sycl_cts;
using namespace get_spec_const;

template <typename T, typename via_kb>
class kernel;

template <typename T, typename via_kb>
class check_specialization_constants_same_name_stress_for_type {
  // Aliases for spec constants
  static constexpr auto &sc_glob = ::sc_same_name<T>;
  static constexpr auto &sc_out = g_outer::sc_same_name<T>;
  static constexpr auto &sc_out_in = g_outer::g_inner::sc_same_name<T>;
  static constexpr auto &sc_out_unn = g_outer::sc_ref_outer_unnamed<T>;
  static constexpr auto &sc_out_unn_in =
      g_outer::g_u_inner::sc_ref_outer_unnamed_inner<T>;
  static constexpr auto &sc_out_unn_in_unn =
      g_outer::g_u_inner::sc_ref_outer_unnamed_inner_unnamed<T>;
  static constexpr auto &sc_unn = sc_ref_unnamed<T>;
  static constexpr auto &sc_unn_out = u_outer::sc_ref_unnamed_outer<T>;
  static constexpr auto &sc_unn_out_unn =
      u_outer::sc_ref_unnamed_outer_unnamed<T>;
  static constexpr auto &sc_unn_out_unn_in =
      u_outer::u_inner::sc_ref_unnamed_out_unnamed_inner<T>;

 public:
  void operator()(util::logger &log, const std::string &type_name) {
    using namespace spec_const_help;
    try {
      auto queue = util::get_cts_object::queue();
      const sycl::context ctx = queue.get_context();
      const sycl::device dev = queue.get_device();

      if constexpr (via_kb::value) {
        if (!dev.has(sycl::aspect::online_compiler)) {
          once_per_unit::log(log, "Device does not support online compilation");
          return;
        }
      }
      // Size for result arrays
      constexpr auto size = static_cast<size_t>(sc_st_id::SIZE);
      sycl::range<1> range(size);
      // Using malloc to not initialize for struct with no default constructor
      // Array of expected default values
      util::remove_initialization<T> ref_def_values_arr[size] {};
      // Array of expected values
      util::remove_initialization<T> ref_arr[size] {};
      // Array for real default values
      util::remove_initialization<T> def_values_arr[size] {};
      // Array for real values
      util::remove_initialization<T> result_arr[size] {};

      // Initialize ref arrays
      for (int i = 0; i < size; ++i) {
        fill_init_values(ref_def_values_arr[i].value, i);
        fill_init_values(ref_arr[i].value, i + static_cast<int>(size));
      }

      {
        sycl::buffer<T, 1> result_buffer(result_arr->data(), range);
        queue.submit([&](sycl::handler &cgh) {
          // Kernel name
          using kernel_name = kernel<T, via_kb>;
          using TargetT = typename std::conditional<
              via_kb::value, sycl::kernel_bundle<sycl::bundle_state::input>,
              sycl::handler>::type;

          // Get default values and set new ones for spec consts
          auto get_default_and_set = [&](TargetT &tgt) {
            // Getting default values of all spec constants
            def_values_arr[0] =
                tgt.template get_specialization_constant<sc_glob>();
            def_values_arr[1] =
                tgt.template get_specialization_constant<sc_out>();
            def_values_arr[2] =
                tgt.template get_specialization_constant<sc_out_in>();
            def_values_arr[3] =
                tgt.template get_specialization_constant<sc_out_unn>();
            def_values_arr[4] =
                tgt.template get_specialization_constant<sc_out_unn_in>();
            def_values_arr[5] =
                tgt.template get_specialization_constant<sc_out_unn_in_unn>();
            def_values_arr[6] =
                tgt.template get_specialization_constant<sc_unn>();
            def_values_arr[7] =
                tgt.template get_specialization_constant<sc_unn_out>();
            def_values_arr[8] =
                tgt.template get_specialization_constant<sc_unn_out_unn>();
            def_values_arr[9] =
                tgt.template get_specialization_constant<sc_unn_out_unn_in>();

            // Setting values to all spec constants
            tgt.template set_specialization_constant<sc_glob>(ref_arr[0]);
            tgt.template set_specialization_constant<sc_out>(ref_arr[1]);
            tgt.template set_specialization_constant<sc_out_in>(ref_arr[2]);
            tgt.template set_specialization_constant<sc_out_unn>(ref_arr[3]);
            tgt.template set_specialization_constant<sc_out_unn_in>(ref_arr[4]);
            tgt.template set_specialization_constant<sc_out_unn_in_unn>(
                ref_arr[5]);
            tgt.template set_specialization_constant<sc_unn>(ref_arr[6]);
            tgt.template set_specialization_constant<sc_unn_out>(ref_arr[7]);
            tgt.template set_specialization_constant<sc_unn_out_unn>(
                ref_arr[8]);
            tgt.template set_specialization_constant<sc_unn_out_unn_in>(
                ref_arr[9]);
          };

          auto res_acc =
              result_buffer.template get_access<sycl::access_mode::write>(cgh);
          // Read all spec consts values from kernel_handler
          auto read_from_kernel_handler = [&]() {
            cgh.single_task<kernel_name>([=](sycl::kernel_handler h) {
              res_acc[0] = h.get_specialization_constant<sc_glob>();
              res_acc[1] = h.get_specialization_constant<sc_out>();
              res_acc[2] = h.get_specialization_constant<sc_out_in>();
              res_acc[3] = h.get_specialization_constant<sc_out_unn>();
              res_acc[4] = h.get_specialization_constant<sc_out_unn_in>();
              res_acc[5] = h.get_specialization_constant<sc_out_unn_in_unn>();
              res_acc[6] = h.get_specialization_constant<sc_unn>();
              res_acc[7] = h.get_specialization_constant<sc_unn_out>();
              res_acc[8] = h.get_specialization_constant<sc_unn_out_unn>();
              res_acc[9] = h.get_specialization_constant<sc_unn_out_unn_in>();
            });
          };
          // No kernel_bundle
          if constexpr (!via_kb::value) {
            get_default_and_set(cgh);
            read_from_kernel_handler();
          } else {
            // Via kernel_bundle
            auto kernelId = sycl::get_kernel_id<kernel_name>();
            auto k_bundle = sycl::get_kernel_bundle<sycl::bundle_state::input>(
                ctx, {dev}, {kernelId});
            if (!k_bundle.has_kernel(kernelId)) {
              log.note("kernel_bundle doesn't contain target kernel for " +
                       type_name_string<T>::get(type_name) + " (skipped)");
              return;
            }
            get_default_and_set(k_bundle);
            auto exec_bundle = sycl::build(k_bundle);
            cgh.use_kernel_bundle(exec_bundle);
            read_from_kernel_handler();
          }
        });
      }
      for (int i = 0; i < size; ++i) {
        if (!check_equal_values(def_values_arr[i], ref_def_values_arr[i])) {
          FAIL(log, "Wrong default value for spec const defined in " +
                        get_hint(i) + " for type " + type_name);
        }
        if (!check_equal_values(result_arr[i], ref_arr[i])) {
          FAIL(log, "Wrong result value for spec const defined in " +
                        get_hint(i) + "for type " + type_name);
        }
      }
    } catch (...) {
      std::string message{"for type " + type_name_string<T>::get(type_name)};
      log.note(message);
      throw;
    }
  }
};

template <typename via_kb>
static void sc_run_test_core(util::logger &log) {
  using namespace specialization_constants_same_name_stress;
  try {
#ifndef SYCL_CTS_FULL_CONFORMANCE
    for_all_types<check_specialization_constants_same_name_stress_for_type,
                  via_kb>(get_spec_const::testing_types::types, log);
#else
    for_all_types_vectors_marray<
        check_specialization_constants_same_name_stress_for_type, via_kb>(
        get_spec_const::testing_types::types, log);
#endif
    for_all_types<check_specialization_constants_same_name_stress_for_type,
                  via_kb>(get_spec_const::testing_types::composite_types, log);

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
  using namespace specialization_constants_same_name_stress;
  try {
    auto queue = util::get_cts_object::queue();
    if (!queue.get_device().has(sycl::aspect::fp16)) {
      log.note(
          "Device does not support half precision floating point "
          "operations");
      return;
    }
#ifndef SYCL_CTS_FULL_CONFORMANCE
    check_specialization_constants_same_name_stress_for_type<sycl::half, via_kb>
        fp16_test{};
    fp16_test(log, "sycl::half");
#else
    for_type_vectors_marray<
        check_specialization_constants_same_name_stress_for_type, sycl::half,
        via_kb>(log, "sycl::half");
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
  using namespace specialization_constants_same_name_stress;
  try {
    auto queue = util::get_cts_object::queue();
    if (!queue.get_device().has(sycl::aspect::fp64)) {
      log.note(
          "Device does not support double precision floating point "
          "operations");
      return;
    }
#ifndef SYCL_CTS_FULL_CONFORMANCE
    check_specialization_constants_same_name_stress_for_type<double, via_kb>
        fp64_test{};
    fp64_test(log, "double");
#else
    for_type_vectors_marray<
        check_specialization_constants_same_name_stress_for_type, double,
        via_kb>(log, "double");
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

}  // namespace specialization_constants_same_name_stress

#endif  // __SYCLCTS_TESTS_SPEC_CONST_SAME_NAME_STRESS_H

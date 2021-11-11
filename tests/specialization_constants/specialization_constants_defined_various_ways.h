/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Common checks for specialization constants defined in various ways
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_SPEC_CONST_DEFINED_VARIOUS_WAYS_H
#define __SYCLCTS_TESTS_SPEC_CONST_DEFINED_VARIOUS_WAYS_H

#include "../common/common.h"
#include "../common/once_per_unit.h"
#include "specialization_constants_common.h"
#include "specialization_constants_defined_various_ways_helper.h"

namespace specialization_constants_defined_various_ways {
using namespace sycl_cts;
using namespace get_spec_const;

template <typename T, int num_case, typename via_kb>
class kernel;

template <typename T, int case_num, typename via_kb, auto &SpecConst>
void perform_test(util::logger &log, const std::string &type_name,
                  const std::string &case_hint) {
  auto queue = util::get_cts_object::queue();
  const sycl::context ctx = queue.get_context();
  const sycl::device dev = queue.get_device();

  if constexpr (via_kb::value) {
    if (!dev.has(sycl::aspect::online_compiler)) {
      once_per_unit::log(log, "Device does not support online compilation");
      return;
    }
  }
  sycl::range<1> range(1);
  T result { get_init_value_helper<T>(0) };
  T ref { get_init_value_helper<T>(0) };
  {
    fill_init_values(ref, case_num);
    sycl::buffer<T, 1> result_buffer(&result, range);
    queue.submit([&](sycl::handler &cgh) {
      auto res_acc =
          result_buffer.template get_access<sycl::access_mode::write>(cgh);
      // Via kernel_bundle
      if constexpr (via_kb::value) {
        auto kernelId = sycl::get_kernel_id<kernel<T, case_num, via_kb>>();
        sycl::kernel_bundle k_bundle =
            sycl::get_kernel_bundle<sycl::bundle_state::input>(ctx, {dev},
                                                               {kernelId});
        if (!k_bundle.has_kernel(kernelId)) {
          log.note("kernel_bundle doesn't contain target kernel in case (" +
                   case_hint + ") for " + type_name_string<T>::get(type_name) +
                   " (skipped)");
          return;
        }
        k_bundle.template set_specialization_constant<SpecConst>(ref);
        auto exec_bundle = sycl::build(k_bundle);
        cgh.use_kernel_bundle(exec_bundle);
        cgh.single_task<kernel<T, case_num, via_kb>>(
            [=](sycl::kernel_handler h) {
              res_acc[0] = h.get_specialization_constant<SpecConst>();
            });
      } else {
        // No kernel_bundle
        cgh.set_specialization_constant<SpecConst>(ref);
        cgh.single_task<kernel<T, case_num, via_kb>>(
            [=](sycl::kernel_handler h) {
              res_acc[0] = h.get_specialization_constant<SpecConst>();
            });
      }
    });
  }
  if (!check_equal_values(ref, result))
    FAIL(log,
         "case (" + case_hint + ") for " + type_name_string<T>::get(type_name));
}

template <typename T, typename via_kb>
class check_specialization_constants_defined_various_ways_for_type {
 public:
  void operator()(util::logger &log, const std::string &type_name) {
    namespace sch = spec_const_help;

    // Case: defined in a non-global named namespace
    {
      constexpr sch::sc_vw_id test_id = sch::sc_vw_id::nonglob;
      constexpr int case_num = to_integral(test_id);
      perform_test<T, case_num, via_kb, sch::sc_nonglob<T, case_num>>(
          log, type_name, get_hint(test_id));
    }

    // Case: defined in an unnamed namespace
    {
      constexpr sch::sc_vw_id test_id = sch::sc_vw_id::unnamed;
      constexpr int case_num = to_integral(test_id);
      perform_test<T, case_num, via_kb, sc_unnamed<T, case_num>>(
          log, type_name, get_hint(test_id));
    }

    // Case: defined in the global namespace as inline constexpr
    {
      constexpr sch::sc_vw_id test_id = sch::sc_vw_id::glob_inl;
      constexpr int case_num = to_integral(test_id);
      perform_test<T, case_num, via_kb, sc_glob_inl<T, case_num>>(
          log, type_name, get_hint(test_id));
    }

    // Case: defined in the global namespace as static constexpr
    {
      constexpr sch::sc_vw_id test_id = sch::sc_vw_id::glob_static;
      constexpr int case_num = to_integral(test_id);
      perform_test<T, case_num, via_kb, sc_glob_static<T, case_num>>(
          log, type_name, get_hint(test_id));
    }

    // Case: a static member variable of a struct in the global namespace
    {
      constexpr sch::sc_vw_id test_id = sch::sc_vw_id::str_glob;
      constexpr int case_num = to_integral(test_id);
      perform_test<T, case_num, via_kb, struct_glob::sc<T, case_num>>(
          log, type_name, get_hint(test_id));
    }

    // Case: a static member variable of a struct in a non-global namespace
    {
      constexpr sch::sc_vw_id test_id = sch::sc_vw_id::str_nonglob;
      constexpr int case_num = to_integral(test_id);
      perform_test<T, case_num, via_kb, sch::struct_nonglob::sc<T, case_num>>(
          log, type_name, get_hint(test_id));
    }

    // Case: a static member variable of a struct in an unnamed namespace
    {
      constexpr sch::sc_vw_id test_id = sch::sc_vw_id::str_unnamed;
      constexpr int case_num = to_integral(test_id);
      perform_test<T, case_num, via_kb, struct_unnamed::sc<T, case_num>>(
          log, type_name, get_hint(test_id));
    }

    // Case: a static member variable declared inline constexpr of a struct in
    // the global namespace
    {
      constexpr sch::sc_vw_id test_id = sch::sc_vw_id::str_glob_inl;
      constexpr int case_num = to_integral(test_id);
      perform_test<T, case_num, via_kb, struct_glob_inl::sc<T, case_num>>(
          log, type_name, get_hint(test_id));
    }

    // Case: a static member variable of a templated struct in the global
    // namespace
    {
      constexpr sch::sc_vw_id test_id = sch::sc_vw_id::str_glob_tmpl;
      constexpr int case_num = to_integral(test_id);
      perform_test<T, case_num, via_kb,
                   struct_glob_tmpl<T>::template sc<case_num>>(
          log, type_name, get_hint(test_id));
    }
  }
};

template <typename via_kb>
static void sc_run_test_core(util::logger &log) {
  using namespace specialization_constants_defined_various_ways;
  {
#ifndef SYCL_CTS_ENABLE_FULL_CONFORMANCE
    for_all_types<check_specialization_constants_defined_various_ways_for_type,
                  via_kb>(get_spec_const::testing_types::types, log);
#else
    for_all_types_vectors_marray<
        check_specialization_constants_defined_various_ways_for_type, via_kb>(
        get_spec_const::testing_types::types, log);
#endif
    for_all_types<check_specialization_constants_defined_various_ways_for_type,
                  via_kb>(get_spec_const::testing_types::composite_types, log);
  }
}

template <typename via_kb>
static void sc_run_test_fp16(util::logger &log) {
  using namespace specialization_constants_defined_various_ways;
  {
    auto queue = util::get_cts_object::queue();
    if (!queue.get_device().has(sycl::aspect::fp16)) {
      log.note(
          "Device does not support half precision floating point "
          "operations");
      return;
    }
#ifndef SYCL_CTS_ENABLE_FULL_CONFORMANCE
    check_specialization_constants_defined_various_ways_for_type<sycl::half,
                                                                 via_kb>
        fp16_test{};
    fp16_test(log, "sycl::half");
#else
    for_type_vectors_marray<
        check_specialization_constants_defined_various_ways_for_type,
        sycl::half, via_kb>(log, "sycl::half");
#endif
  }
}

template <typename via_kb>
static void sc_run_test_fp64(util::logger &log) {
  using namespace specialization_constants_defined_various_ways;
  {
    auto queue = util::get_cts_object::queue();
    if (!queue.get_device().has(sycl::aspect::fp64)) {
      log.note(
          "Device does not support double precision floating point "
          "operations");
      return;
    }
#ifndef SYCL_CTS_ENABLE_FULL_CONFORMANCE
    check_specialization_constants_defined_various_ways_for_type<double, via_kb>
        fp64_test{};
    fp64_test(log, "double");
#else
    for_type_vectors_marray<
        check_specialization_constants_defined_various_ways_for_type, double,
        via_kb>(log, "double");
#endif
  }
}

}  // namespace specialization_constants_defined_various_ways

#endif  // __SYCLCTS_TESTS_SPEC_CONST_DEFINED_VARIOUS_WAYS_H

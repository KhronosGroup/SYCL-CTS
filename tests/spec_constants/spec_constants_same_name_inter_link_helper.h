/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides sycl::specialization_id definitions for specialization constants
//  with same name and internal linkage
//
*******************************************************************************/
#ifndef __SYCLCTS_TESTS_SPEC_CONST_SAME_NAME_INTERNAL_LINK_HELPER_H
#define __SYCLCTS_TESTS_SPEC_CONST_SAME_NAME_INTERNAL_LINK_HELPER_H

#include "../common/common.h"
#include "spec_constants_common.h"

namespace spec_const_help {

template <int tu_num>
struct sc_sn_il_config {
  static constexpr int tu = tu_num;
  static constexpr int ref_val = tu_num + 1;
};

}  // namespace spec_const_help

//  Using unnamed namespace to make it static and enforce internal linkage in
//  accordance with the test plan, taking into account that it is necessary
//  to perform a test for several types
namespace {

// SC_SN_IL_TU_NUM is defined in every TU
template <typename T, typename via_kb>
constexpr sycl::specialization_id<T> sc_same_name_inter_link(
    user_def_types::get_init_value<T>(SC_SN_IL_TU_NUM));

}  // namespace

#endif  // __SYCLCTS_TESTS_SPEC_CONST_SAME_NAME_INTERNAL_LINK_HELPER_H

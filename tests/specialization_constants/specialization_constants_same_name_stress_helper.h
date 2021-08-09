/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides sycl::specialization_id definitions for specialization constants
//  same name stress tests
//
*******************************************************************************/
#ifndef __SYCLCTS_TESTS_SPEC_CONST_SAME_NAME_STRESS_HELPER_H
#define __SYCLCTS_TESTS_SPEC_CONST_SAME_NAME_STRESS_HELPER_H

#include <map>

#include "specialization_constants_common.h"

namespace gsc = get_spec_const;

namespace spec_const_help {

enum class sc_st_id : int {
  glob,
  outer,
  outer_inner,
  outer_unnamed,
  outer_unnamed_inner,
  outer_unnamed_inner_unnamed,
  unnamed,
  unnamed_outer,
  unnamed_outer_unnamed,
  unnamed_outer_unnamed_inner,
  SIZE  // This should be last
};

static std::string get_hint(int test_id) {
  static const std::map<int, std::string> sc_st_tests_hints{
      {to_integral(sc_st_id::glob), "global namespace"},
      {to_integral(sc_st_id::outer), "outer namespace"},
      {to_integral(sc_st_id::outer_inner), "outer::inner namespace"},
      {to_integral(sc_st_id::outer_unnamed), "outer::unnamed namespace"},
      {to_integral(sc_st_id::outer_unnamed_inner),
       "outer::unnamed::inner namespace"},
      {to_integral(sc_st_id::outer_unnamed_inner_unnamed),
       "outer::unnamed::inner::unnamed namespace"},
      {to_integral(sc_st_id::unnamed), "unnamed namespace"},
      {to_integral(sc_st_id::unnamed_outer), "unnamed::outer namespace"},
      {to_integral(sc_st_id::unnamed_outer_unnamed),
       "unnamed::outer::unnamed namespace"},
      {to_integral(sc_st_id::unnamed_outer_unnamed_inner),
       "unnamed::outer::unnamed::inner namespace"}};

  return sc_st_tests_hints.at(test_id);
}

}  // namespace spec_const_help

// Specialization constant defined in global namespace
template <typename T>
constexpr sycl::specialization_id<T> sc_same_name(gsc::get_init_value_helper<T>(
    to_integral(spec_const_help::sc_st_id::glob)));

namespace g_outer {
// Specialization constant defined in outer namespace
template <typename T>
constexpr sycl::specialization_id<T> sc_same_name(gsc::get_init_value_helper<T>(
    to_integral(spec_const_help::sc_st_id::outer)));

namespace g_inner {
// Specialization constant defined in outer::inner namespace
template <typename T>
constexpr sycl::specialization_id<T> sc_same_name(gsc::get_init_value_helper<T>(
    to_integral(spec_const_help::sc_st_id::outer_inner)));
}  // namespace g_inner

namespace {
// Specialization constant defined in outer::unnamed namespace
template <typename T>
constexpr sycl::specialization_id<T> sc_same_name(gsc::get_init_value_helper<T>(
    to_integral(spec_const_help::sc_st_id::outer_unnamed)));

template <typename T>
constexpr auto& sc_ref_outer_unnamed = sc_same_name<T>;

namespace g_u_inner {
// Specialization constant defined in outer::unnamed::inner namespace
template <typename T>
constexpr sycl::specialization_id<T> sc_same_name(gsc::get_init_value_helper<T>(
    to_integral(spec_const_help::sc_st_id::outer_unnamed_inner)));

template <typename T>
constexpr auto& sc_ref_outer_unnamed_inner = sc_same_name<T>;

namespace {
// Specialization constant defined in outer::unnamed::inner::unnamed
// namespace
template <typename T>
constexpr sycl::specialization_id<T> sc_same_name(
    gsc::get_init_value_helper<T>(
        to_integral(spec_const_help::sc_st_id::outer_unnamed_inner_unnamed)));

template <typename T>
constexpr auto& sc_ref_outer_unnamed_inner_unnamed = sc_same_name<T>;
}  // unnamed namespace
}  // namespace g_u_inner
}  // unnamed namespace
}  // namespace g_outer

namespace {
// Specialization constant defined in unnamed namespace
template <typename T>
constexpr sycl::specialization_id<T> sc_same_name(gsc::get_init_value_helper<T>(
    to_integral(spec_const_help::sc_st_id::unnamed)));

template <typename T>
constexpr auto& sc_ref_unnamed = sc_same_name<T>;

namespace u_outer {
// Specialization constant defined in unnamed::outer namespace
template <typename T>
constexpr sycl::specialization_id<T> sc_same_name(gsc::get_init_value_helper<T>(
    to_integral(spec_const_help::sc_st_id::unnamed_outer)));

template <typename T>
constexpr auto& sc_ref_unnamed_outer = sc_same_name<T>;

namespace {
// Specialization constant defined in unnamed::outer::unnamed namespace
template <typename T>
constexpr sycl::specialization_id<T> sc_same_name(gsc::get_init_value_helper<T>(
    to_integral(spec_const_help::sc_st_id::unnamed_outer_unnamed)));

template <typename T>
constexpr auto& sc_ref_unnamed_outer_unnamed = sc_same_name<T>;

namespace u_inner {
// Specialization constant defined in unnamed::outer::unnamed::inner
// namespace
template <typename T>
constexpr sycl::specialization_id<T> sc_same_name(gsc::get_init_value_helper<T>(
    to_integral(spec_const_help::sc_st_id::unnamed_outer_unnamed_inner)));

template <typename T>
constexpr auto& sc_ref_unnamed_out_unnamed_inner = sc_same_name<T>;
}  // namespace u_inner
}  // unnamed namespace
}  // namespace u_outer
}  // unnamed namespace

#endif  // __SYCLCTS_TESTS_SPEC_CONST_SAME_NAME_STRESS_HELPER_H

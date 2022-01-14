/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides sycl::specialization_id definitions for specialization constants
//  defined various ways tests
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_SPEC_CONST_DEFINED_VARIOUS_WAYS_HELPER_H
#define __SYCLCTS_TESTS_SPEC_CONST_DEFINED_VARIOUS_WAYS_HELPER_H

#include <map>

#include "specialization_constants_common.h"

namespace gsc = get_spec_const;

namespace spec_const_help {

enum class sc_vw_id : int {
  nonglob = 1,
  unnamed,
  glob_inl,
  glob_static,
  str_glob,
  str_nonglob,
  str_unnamed,
  str_glob_inl,
  str_glob_tmpl
};

static std::string get_hint(sc_vw_id test_id) {
  static const std::map<sc_vw_id, std::string> sc_vw_test_hints{
      {sc_vw_id::nonglob, "defined in a non-global named namespace"},
      {sc_vw_id::unnamed, "defined in an unnamed namespace"},
      {sc_vw_id::glob_inl,
       "defined in the global namespace as inline constexpr"},
      {sc_vw_id::glob_static,
       "defined in the global namespace as static constexpr"},
      {sc_vw_id::str_glob,
       "a static member variable of a struct in the global namespace"},
      {sc_vw_id::str_nonglob,
       "a static member variable of a struct in a non-global namespace"},
      {sc_vw_id::str_unnamed,
       "a static member variable of a struct in an unnamed namespace"},
      {sc_vw_id::str_glob_inl,
       "a static member variable declared inline constexpr of a struct in the "
       "global namespace"},
      {sc_vw_id::str_glob_tmpl,
       "a static member variable of a templated struct in the global "
       "namespace"}};
  return sc_vw_test_hints.at(test_id);
}

// Defined in a non-global named namespace
template <typename T, int case_num>
constexpr sycl::specialization_id<T> sc_nonglob(
    user_def_types::get_init_value_helper<T>(case_num));

// A static member variable of a struct in a non-global namespace
struct struct_nonglob {
  constexpr struct_nonglob() {}
  template <typename T, int case_num>
  static constexpr sycl::specialization_id<T> sc{
      user_def_types::get_init_value_helper<T>(case_num)};
};
}  // namespace spec_const_help

namespace {
// Defined in an unnamed namespace
template <typename T, int case_num>
constexpr sycl::specialization_id<T> sc_unnamed(
    user_def_types::get_init_value_helper<T>(case_num));

// A static member variable of a struct in an unnamed namespace
struct struct_unnamed {
  constexpr struct_unnamed() {}
  template <typename T, int case_num>
  static constexpr sycl::specialization_id<T> sc{
      user_def_types::get_init_value_helper<T>(case_num)};
};
}  // unnamed namespace

// Defined in the global namespace as inline constexpr
template <typename T, int case_num>
inline constexpr sycl::specialization_id<T> sc_glob_inl(
    user_def_types::get_init_value_helper<T>(case_num));

// Defined in the global namespace as static constexpr
template <typename T, int case_num>
static constexpr sycl::specialization_id<T> sc_glob_static(
    user_def_types::get_init_value_helper<T>(case_num));

// A static member variable of a struct in the global namespace
struct struct_glob {
  constexpr struct_glob() {}
  template <typename T, int case_num>
  static constexpr sycl::specialization_id<T> sc{
      user_def_types::get_init_value_helper<T>(case_num)};
};

// A static member variable declared inline constexpr of a struct in the global
// namespace
struct struct_glob_inl {
  constexpr struct_glob_inl() {}
  template <typename T, int case_num>
  static inline constexpr sycl::specialization_id<T> sc{
      user_def_types::get_init_value_helper<T>(case_num)};
};

// A static member variable of a templated struct in the global namespace
template <typename T>
struct struct_glob_tmpl {
  constexpr struct_glob_tmpl() {}
  template <int case_num>
  static constexpr sycl::specialization_id<T> sc{
      user_def_types::get_init_value_helper<T>(case_num)};
};

#endif  // __SYCLCTS_TESTS_SPEC_CONST_DEFINED_VARIOUS_WAYS_HELPER_H

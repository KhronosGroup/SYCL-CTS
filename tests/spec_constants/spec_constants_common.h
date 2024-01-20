/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides common type list for specialization constants type coverage
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_SPEC_CONST_COMMON_H
#define __SYCLCTS_TESTS_SPEC_CONST_COMMON_H

#include "../common/common.h"
#include "../common/type_coverage.h"
#include "../common/type_list.h"

#include <numeric>

namespace get_spec_const {
namespace testing_types {

// C++ fundamental types that will be used in type coverage
static const auto types = named_type_pack<
    bool, char, signed char, unsigned char, short, unsigned short, int,
    unsigned int, long, unsigned long, long long, unsigned long long, float,
    sycl::byte, std::int8_t, std::uint8_t, std::int16_t, std::uint16_t,
    std::int32_t, std::uint32_t, std::int64_t, std::uint64_t, std::size_t>::generate(
    "bool",         "char",           "signed char",  "unsigned char",
    "short",        "unsigned short", "int",          "unsigned int",
    "long",         "unsigned long",  "long long",    "unsigned long long",
    "float",        "sycl::byte",     "std::int8_t",  "std::uint8_t",
    "std::int16_t", "std::uint16_t",  "std::int32_t", "std::uint32_t",
    "std::int64_t", "std::uint64_t",  "std::size_t");

// custom data types that will be used in type coverage
static const auto composite_types =
    named_type_pack<user_def_types::no_cnstr, user_def_types::def_cnstr,
                    user_def_types::no_def_cnstr>::generate(
        "no_cnstr", "def_cnstr", "no_def_cnstr");

}  // namespace testing_types

// Flags that used to specify test type (test with kernel bundle or without
// kernel bundle)
using sc_use_kernel_bundle = std::true_type;
using sc_no_kernel_bundle = std::false_type;

constexpr int default_val = 20;

template <typename T, int case_num>
constexpr sycl::specialization_id<T> spec_const(
    user_def_types::get_init_value<T>(default_val));

template <typename T>
void fill_init_values(T &result, int val) {
  result = user_def_types::get_init_value<T>(val);
}

template <typename T, int numElements>
void fill_init_values(sycl::vec<T, numElements> &result, int val) {
  // Fill manually because sycl::vec does not have iterators
  for (int i = 0; i < numElements; ++i) {
    result[i] = val;
  }
}

template <typename T, std::size_t numElements>
void fill_init_values(sycl::marray<T, numElements> &result, int val) {
  std::fill(result.begin(), result.end(), val);
}

enum class test_cases_external {
  by_reference_via_handler = 1,
  by_value_via_handler = 2,
  by_reference_via_bundle = 3,
  by_value_via_bundle = 4
};

}  // namespace get_spec_const

#define SINGLE_ARG(...) __VA_ARGS__
#define SYCL_VECTORS_MARRAYS(TYPE, FUNC)  \
  FUNC(TYPE)                              \
  FUNC(SINGLE_ARG(sycl::vec<TYPE, 1>))    \
  FUNC(SINGLE_ARG(sycl::vec<TYPE, 2>))    \
  FUNC(SINGLE_ARG(sycl::vec<TYPE, 3>))    \
  FUNC(SINGLE_ARG(sycl::vec<TYPE, 4>))    \
  FUNC(SINGLE_ARG(sycl::vec<TYPE, 8>))    \
  FUNC(SINGLE_ARG(sycl::vec<TYPE, 16>))   \
  FUNC(SINGLE_ARG(sycl::marray<TYPE, 2>)) \
  FUNC(SINGLE_ARG(sycl::marray<TYPE, 5>)) \
  FUNC(SINGLE_ARG(sycl::marray<TYPE, 10>))

#define CORE_TYPES(FUNC)   \
  FUNC(bool)               \
  FUNC(char)               \
  FUNC(signed char)        \
  FUNC(unsigned char)      \
  FUNC(short)              \
  FUNC(unsigned short)     \
  FUNC(int)                \
  FUNC(unsigned int)       \
  FUNC(long)               \
  FUNC(unsigned long)      \
  FUNC(long long)          \
  FUNC(unsigned long long) \
  FUNC(float)

#define CORE_TYPES_PARAM(FUNC, X) \
  FUNC(bool, X)                   \
  FUNC(char, X)                   \
  FUNC(signed char, X)            \
  FUNC(unsigned char, X)          \
  FUNC(short, X)                  \
  FUNC(unsigned short, X)         \
  FUNC(int, X)                    \
  FUNC(unsigned int, X)           \
  FUNC(long, X)                   \
  FUNC(unsigned long, X)          \
  FUNC(long long, X)              \
  FUNC(unsigned long long, X)     \
  FUNC(float, X)

#endif  // __SYCLCTS_TESTS_SPEC_CONST_COMMON_H

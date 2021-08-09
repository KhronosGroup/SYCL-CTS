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

#include <numeric>

namespace get_spec_const {
namespace testing_types {

// A user-defined struct with several scalar member variables, no constructor,
//  destructor or member functions.
struct no_cnstr {
  float a;
  int b;
  char c;

  friend bool operator==(const no_cnstr &lhs, const no_cnstr &rhs) {
    return ((lhs.a == rhs.a) && (lhs.b == rhs.b) && (lhs.c == rhs.c));
  }
};

// A user-defined class with several scalar member variables, a user-defined
//  default constructor, and some member functions that modify the member
//  variables.
struct def_cnstr {
  float a;
  int b;
  char c;

 public:
  constexpr def_cnstr() : a(3.0), b(2), c('c') {}

  constexpr void assign(int val) {
    a = val * 3.0;
    b = val * 2;
    c = val;
  }

  inline friend bool operator==(const def_cnstr &lhs, const def_cnstr &rhs) {
    return ((lhs.a == rhs.a) && (lhs.b == rhs.b) && (lhs.c == rhs.c));
  }
};

// A user-defined class with several scalar member variables, a deleted default
// constructor, and a user-defined (non-default) constructor.
class no_def_cnstr {
  float a;
  int b;
  char c;

 public:
  no_def_cnstr() = delete;

  constexpr no_def_cnstr(int val) : a(val * 3.0), b(val * 2), c(val) {}

  friend bool operator==(const no_def_cnstr &lhs, const no_def_cnstr &rhs) {
    return ((lhs.a == rhs.a) && (lhs.b == rhs.b) && (lhs.c == rhs.c));
  }
};

// C++ fundamental types that will be used in type coverage
static const auto types = named_type_pack<
    bool, char, signed char, unsigned char, short, unsigned short, int,
    unsigned int, long, unsigned long, long long, unsigned long long, float,
    sycl::byte, std::int8_t, std::uint8_t, std::int16_t, std::uint16_t,
    std::int32_t, std::uint32_t, std::int64_t, std::uint64_t, std::size_t>{
    "bool",         "char",           "signed char",  "unsigned char",
    "short",        "unsigned short", "int",          "unsigned int",
    "long",         "unsigned long",  "long long",    "unsigned long long",
    "float",        "sycl::byte",     "std::int8_t",  "std::uint8_t",
    "std::int16_t", "std::uint16_t",  "std::int32_t", "std::uint32_t",
    "std::int64_t", "std::uint64_t",  "std::size_t"};

// custom data types that will be used in type coverage
static const auto composite_types =
    named_type_pack<no_cnstr, def_cnstr, no_def_cnstr>(
        {"no_cnstr", "def_cnstr", "no_def_cnstr"});

}  // namespace testing_types

struct sc_use_kernel_bundle {
  static constexpr bool value = true;
};
struct sc_no_kernel_bundle {
  static constexpr bool value = false;
};

template <typename T>
inline constexpr auto get_init_value_helper(int x) {
  return x;
}

template <>
inline constexpr auto get_init_value_helper<testing_types::no_cnstr>(int x) {
  testing_types::no_cnstr instance{};
  instance.a = x;
  instance.b = x;
  instance.c = x;
  return instance;
}

template <>
inline constexpr auto get_init_value_helper<testing_types::def_cnstr>(int x) {
  testing_types::def_cnstr instance;
  instance.assign(x);
  return instance;
}

constexpr int default_val = 20;

template <typename T, int case_num>
constexpr sycl::specialization_id<T> spec_const(
    get_init_value_helper<T>(default_val));

template <typename T>
void fill_init_values(T &result, int val) {
  result = get_init_value_helper<T>(val);
}

template <typename T, int numElements>
void fill_init_values(sycl::vec<T, numElements> &result, int val) {
  // Fill manually because sycl::vec does not have iterators
  for (int i = 0; i < numElements; ++i) {
    result[i] = val + i;
  }
}

template <typename T, std::size_t numElements>
void fill_init_values(sycl::marray<T, numElements> &result, int val) {
  std::iota(result.begin(), result.end(), val);
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

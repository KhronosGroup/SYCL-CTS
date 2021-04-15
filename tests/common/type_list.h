/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
// Provide common type list for type coverage
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_COMMON_TYPE_LIST_H
#define __SYCLCTS_TESTS_COMMON_TYPE_LIST_H

#include "../common/type_coverage.h"

/** user defined struct that is used in accessor tests
*/
struct user_struct {
  float a;
  int b;
  char c;

  using element_type = int;

  user_struct() : a(0), b(0), c(0) {}

  user_struct(int val) : a(0), b(val), c(0) {}

  element_type operator[](size_t index) const { return b; }

  friend bool operator==(const user_struct &lhs, const user_struct &rhs) {
    static constexpr auto eps = 1e-4f;
    return (((lhs.a + eps > rhs.a) && (lhs.a < rhs.a + eps)) &&
            (lhs.b == rhs.b) && (lhs.c == rhs.c));
  }
};

namespace get_cts_types {
static const auto vector_types = named_type_pack<
    bool, char, signed char, unsigned char, short, unsigned short, int,
    unsigned int, long, unsigned long, long long, unsigned long long, float,
    cl::sycl::cl_float, cl::sycl::byte, cl::sycl::cl_bool, cl::sycl::cl_char,
    cl::sycl::cl_uchar, cl::sycl::cl_short, cl::sycl::cl_ushort,
    cl::sycl::cl_int, cl::sycl::cl_uint, cl::sycl::cl_long, cl::sycl::cl_ulong>{
    "bool",
    "char",
    "signed char",
    "unsigned char",
    "short",
    "unsigned short",
    "int",
    "unsigned int",
    "long",
    "unsigned long",
    "long long",
    "unsigned long long",
    "float",
    "cl::sycl::cl_float",
    "cl::sycl::byte",
    "cl::sycl::cl_bool",
    "cl::sycl::cl_char",
    "cl::sycl::cl_uchar",
    "cl::sycl::cl_short",
    "cl::sycl::cl_ushort",
    "cl::sycl::cl_int",
    "cl::sycl::cl_uint",
    "cl::sycl::cl_long",
    "cl::sycl::cl_ulong"};
} // namespace get_cts_type

#endif // __SYCLCTS_TESTS_COMMON_TYPE_LIST_H

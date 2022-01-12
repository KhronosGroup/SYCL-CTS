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

namespace user_def_types {
// A user-defined struct with several scalar member variables, no constructor,
//  destructor or member functions.
struct no_cnstr {
  float a;
  int b;
  char c;

  void operator=(const int &v) {
    this->a = v;
    this->b = v;
    this->c = v;
  }

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

  void operator=(const int &v) {
    this->a = v * 3.0;
    this->b = v * 2;
    this->c = v;
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

  void operator=(const int &v) {
    no_def_cnstr temp(v);
    this->a = temp.a;
    this->b = temp.b;
    this->c = temp.c;
  }
};

// A user-defined struct with several scalar member variables, arrow operator
// overload, no constructor and
// destructor or member functions.
struct arrow_operator_overloaded {
  float a;
  int b;
  char c;

  void operator=(const int &v) {
    this->a = v;
    this->b = v;
    this->c = v;
  }

  arrow_operator_overloaded *operator->() { return this; }
  const arrow_operator_overloaded *operator->() const { return this; }

  friend bool operator==(const arrow_operator_overloaded &lhs,
                         const arrow_operator_overloaded &rhs) {
    return ((lhs.a == rhs.a) && (lhs.b == rhs.b) && (lhs.c == rhs.c));
  }
};

}  // namespace user_def_types

namespace get_cts_types {
static const auto vector_types = named_type_pack<
    bool, char, signed char, unsigned char, short, unsigned short, int,
    unsigned int, long, unsigned long, long long, unsigned long long, float,
    sycl::cl_float, sycl::byte, sycl::cl_bool, sycl::cl_char,
    sycl::cl_uchar, sycl::cl_short, sycl::cl_ushort,
    sycl::cl_int, sycl::cl_uint, sycl::cl_long, sycl::cl_ulong>{
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
    "sycl::cl_float",
    "sycl::byte",
    "sycl::cl_bool",
    "sycl::cl_char",
    "sycl::cl_uchar",
    "sycl::cl_short",
    "sycl::cl_ushort",
    "sycl::cl_int",
    "sycl::cl_uint",
    "sycl::cl_long",
    "sycl::cl_ulong"};
} // namespace get_cts_type

#endif // __SYCLCTS_TESTS_COMMON_TYPE_LIST_H

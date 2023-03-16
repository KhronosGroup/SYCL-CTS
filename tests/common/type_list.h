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
#include "../common/value_operations.h"

/** user defined struct that is used in accessor tests
 */
struct user_struct {
  float a;
  int b;
  char c;

  using element_type = int;

  user_struct() : a(0), b(0), c(0) {}

  user_struct(int val) : a(0), b(val), c(0) {}

  element_type operator[](size_t) const { return b; }

  friend bool operator==(const user_struct &lhs, const user_struct &rhs) {
    static constexpr auto eps = 1e-4f;
    return (((lhs.a + eps > rhs.a) && (lhs.a < rhs.a + eps)) &&
            (lhs.b == rhs.b) && (lhs.c == rhs.c));
  }
  friend bool operator!=(const user_struct &lhs, const user_struct &rhs) {
    return !(lhs == rhs);
  }
};

/** Trivially copyable types that are used in sycl::bit_cast test
 */
template <typename PrimaryType>
class Base {
 public:
  PrimaryType member;
  Base() = default;
  Base(int value) { value_operations::assign(member, value); }
  bool operator==(const Base &rhs) const { return member == rhs.member; }
};

template <typename PrimaryType>
class Derived : public Base<PrimaryType> {
 public:
  Derived() = default;
  Derived(int value) : Base<PrimaryType>(value) {}
};

namespace user_def_types {
// A user-defined struct with several scalar member variables, no constructor,
//  destructor or non-special member functions.
struct no_cnstr {
  float a;
  int b;
  char c;

  constexpr auto &operator=(const int &v) {
    a = v;
    b = v;
    c = v;
    return *this;
  }

  friend bool operator==(const no_cnstr &lhs, const no_cnstr &rhs) {
    return ((lhs.a == rhs.a) && (lhs.b == rhs.b) && (lhs.c == rhs.c));
  }

  friend bool operator!=(const no_cnstr &lhs, const no_cnstr &rhs) {
    return !(lhs == rhs);
  }

  friend no_cnstr operator+(const no_cnstr &lhs, int i) {
    return {lhs.a + static_cast<float>(i), lhs.b + i,
            static_cast<char>(lhs.c + i)};
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
  constexpr def_cnstr() : a(3.0f), b(2), c('c') {}

  constexpr void assign(int val) {
    a = val * 3.0f;
    b = val * 2;
    c = val;
  }

  constexpr auto &operator=(const int &v) {
    assign(v);
    return *this;
  }

  inline friend bool operator==(const def_cnstr &lhs, const def_cnstr &rhs) {
    return ((lhs.a == rhs.a) && (lhs.b == rhs.b) && (lhs.c == rhs.c));
  }

  friend bool operator!=(const def_cnstr &lhs, const def_cnstr &rhs) {
    return !(lhs == rhs);
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

  constexpr no_def_cnstr(int val) : a(val * 3.0f), b(val * 2), c(val) {}

  friend bool operator==(const no_def_cnstr &lhs, const no_def_cnstr &rhs) {
    return ((lhs.a == rhs.a) && (lhs.b == rhs.b) && (lhs.c == rhs.c));
  }

  friend bool operator!=(const no_def_cnstr &lhs, const no_def_cnstr &rhs) {
    return !(lhs == rhs);
  }
};

// A user-defined struct with several scalar member variables, arrow operator
// overload, no constructor and destructor or member functions.
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
  friend bool operator!=(const arrow_operator_overloaded &lhs,
                         const arrow_operator_overloaded &rhs) {
    return !(lhs == rhs);
  }
};

// Returns instance of type T
template <typename T>
inline constexpr auto get_init_value_helper(int x) {
  return x;
}

// Returns instance of type bool
template <>
inline constexpr auto get_init_value_helper<bool>(int x) {
  return (x % 2 != 0);
}

// Returns instance of user defined struct with no constructor
template <>
inline constexpr auto get_init_value_helper<no_cnstr>(int x) {
  no_cnstr instance{};
  instance = x;
  return instance;
}

// Returns instance of user defined struct default constructor
template <>
inline constexpr auto get_init_value_helper<def_cnstr>(int x) {
  def_cnstr instance;
  instance = x;
  return instance;
}

}  // namespace user_def_types

namespace get_cts_types {
inline auto get_vector_types() {
#if SYCL_CTS_ENABLE_OPENCL_INTEROP_TESTS
  static const auto pack = named_type_pack<
      bool, char, signed char, unsigned char, short, unsigned short, int,
      unsigned int, long, unsigned long, long long, unsigned long long, float,
      sycl::cl_float, sycl::byte, sycl::cl_bool, sycl::cl_char, sycl::cl_uchar,
      sycl::cl_short, sycl::cl_ushort, sycl::cl_int, sycl::cl_uint,
      sycl::cl_long, sycl::cl_ulong>::generate("bool", "char", "signed char",
                                               "unsigned char", "short",
                                               "unsigned short", "int",
                                               "unsigned int", "long",
                                               "unsigned long", "long long",
                                               "unsigned long long", "float",
                                               "sycl::cl_float", "sycl::byte",
                                               "sycl::cl_bool", "sycl::cl_char",
                                               "sycl::cl_uchar",
                                               "sycl::cl_short",
                                               "sycl::cl_ushort",
                                               "sycl::cl_int", "sycl::cl_uint",
                                               "sycl::cl_long",
                                               "sycl::cl_ulong");
#else
  static const auto pack =
      named_type_pack<bool, char, signed char, unsigned char, short,
                      unsigned short, int, unsigned int, long, unsigned long,
                      long long, unsigned long long,
                      float>::generate("bool", "char", "signed char",
                                       "unsigned char", "short",
                                       "unsigned short", "int", "unsigned int",
                                       "long", "unsigned long", "long long",
                                       "unsigned long long", "float");
#endif
  return pack;
}

inline auto get_fp64_type() {
  static const auto types = named_type_pack<double>::generate("double");
  return types;
}

}  // namespace get_cts_types

namespace deduction {
#if !SYCL_CTS_ENABLE_FULL_CONFORMANCE
static const auto vector_types =
    named_type_pack<int, float>::generate("int", "float");
#else
static const auto vector_types =
    named_type_pack<char, signed char, unsigned char, short int,
                    unsigned short int, unsigned int, long int,
                    unsigned long int, long long int, unsigned long long int,
                    bool, float>::generate("char", "signed char",
                                           "unsigned char", "short int",
                                           "unsigned short int", "unsigned int",
                                           "long int", "unsigned long int",
                                           "long long int",
                                           "unsigned long long int", "bool",
                                           "float");
#endif

static const auto scalar_types =
    named_type_pack<user_struct, user_def_types::no_cnstr,
                    user_def_types::no_def_cnstr>::generate("user_struct",
                                                            "no_cnstr",
                                                            "no_def_cnstr");
}  // namespace deduction

#endif  // __SYCLCTS_TESTS_COMMON_TYPE_LIST_H

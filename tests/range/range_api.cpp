/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME range_api

namespace range_api__ {
using namespace sycl_cts;

static const size_t expected_a[] = {16, 128, 256};
static const size_t expected_b[] = {2, 16, 128};

template <int dims>
void test_gets(util::logger &log, cl::sycl::range<dims> r) {
  int i;
  for (i = 0; i < dims; i++) {
    CHECK_TYPE(log, r.get(i), expected_a[i]);
    CHECK_VALUE(log, r.get(i), expected_a[i], i);
    CHECK_TYPE(log, r[i], expected_a[i]);
    CHECK_VALUE(log, r[i], expected_a[i], i);
  }
}

template <int dims, typename T>
T test_AND(const T &lhs, const T &rhs) {
  if (lhs && rhs) return 1;
  return 0;
}
template <int dims, typename T>
T test_OR(const T &lhs, const T &rhs) {
  if (lhs || rhs) return 1;
  return 0;
}

template <int dims>
void test_operators(util::logger &log, cl::sycl::range<dims> r_one,
                    cl::sycl::range<dims> r_two) {
  // operators performed
  cl::sycl::range<dims> add = r_one + r_two;
  cl::sycl::range<dims> subtract = r_one - r_two;
  cl::sycl::range<dims> divide = r_one / r_two;
  cl::sycl::range<dims> multiply = r_one * r_two;
  cl::sycl::range<dims> mod = r_one % r_two;
  cl::sycl::range<dims> r_shift = r_one >> r_two;
  cl::sycl::range<dims> l_shift = r_one << r_two;
  cl::sycl::range<dims> and1 = (r_one & r_two);
  cl::sycl::range<dims> or1 = (r_one | r_two);
  cl::sycl::range<dims> xor1 = r_one ^ r_two;
  cl::sycl::range<dims> and_log = r_one && r_two;
  cl::sycl::range<dims> or_log = r_one || r_two;

  size_t a = 16;

  cl::sycl::range<dims> add_size_t_r = r_one + a;
  cl::sycl::range<dims> add_size_t_l = a + r_one;
  cl::sycl::range<dims> subtract_size_t_r = r_one - a;
  cl::sycl::range<dims> subtract_size_t_l = a - r_one;
  cl::sycl::range<dims> divide_size_t_r = r_one / a;
  cl::sycl::range<dims> divide_size_t_l = a / r_one;
  cl::sycl::range<dims> multiply_size_t_r = r_one * a;
  cl::sycl::range<dims> multiply_size_t_l = a * r_one;
  cl::sycl::range<dims> mod_size_t_r = r_one % a;
  cl::sycl::range<dims> mod_size_t_l = a % r_one;
  cl::sycl::range<dims> l_shift_size_t_r = r_one << a;
  cl::sycl::range<dims> l_shift_size_t_l = a << r_one;
  cl::sycl::range<dims> r_shift_size_t_r = r_one >> a;
  cl::sycl::range<dims> r_shift_size_t_l = a >> r_one;

  int i;
  for (i = 0; i < dims; i++) {
    // operator +
    CHECK_TYPE(log, add.get(i), expected_a[i] + expected_b[i]);
    CHECK_VALUE(log, add.get(i), expected_a[i] + expected_b[i], i);
    CHECK_TYPE(log, add[i], expected_a[i] + expected_b[i]);
    CHECK_VALUE(log, add[i], expected_a[i] + expected_b[i], i);

    // operator -
    CHECK_TYPE(log, subtract.get(i), expected_a[i] - expected_b[i]);
    CHECK_VALUE(log, subtract.get(i), expected_a[i] - expected_b[i], i);
    CHECK_TYPE(log, subtract[i], expected_a[i] - expected_b[i]);
    CHECK_VALUE(log, subtract[i], expected_a[i] - expected_b[i], i);

    // operator /
    CHECK_TYPE(log, divide.get(i), expected_a[i] / expected_b[i]);
    CHECK_VALUE(log, divide.get(i), expected_a[i] / expected_b[i], i);
    CHECK_TYPE(log, divide[i], expected_a[i] / expected_b[i]);
    CHECK_VALUE(log, divide[i], expected_a[i] / expected_b[i], i);

    // operator *
    CHECK_TYPE(log, multiply.get(i), expected_a[i] * expected_b[i]);
    CHECK_VALUE(log, multiply.get(i), expected_a[i] * expected_b[i], i);
    CHECK_TYPE(log, multiply[i], expected_a[i] * expected_b[i]);
    CHECK_VALUE(log, multiply[i], expected_a[i] * expected_b[i], i);

    // operator %
    CHECK_TYPE(log, mod.get(i), expected_a[i] % expected_b[i]);
    CHECK_VALUE(log, mod.get(i), expected_a[i] % expected_b[i], i);
    CHECK_TYPE(log, mod[i], expected_a[i] % expected_b[i]);
    CHECK_VALUE(log, mod[i], expected_a[i] % expected_b[i], i);

    // operator >>
    CHECK_TYPE(log, r_shift.get(i), expected_a[i] >> expected_b[i]);
    CHECK_VALUE(log, r_shift.get(i), expected_a[i] >> expected_b[i], i);
    CHECK_TYPE(log, r_shift[i], expected_a[i] >> expected_b[i]);
    CHECK_VALUE(log, r_shift[i], expected_a[i] >> expected_b[i], i);

    // operator <<
    CHECK_TYPE(log, l_shift.get(i), expected_a[i] << expected_b[i]);
    CHECK_VALUE(log, l_shift.get(i), expected_a[i] << expected_b[i], i);
    CHECK_TYPE(log, l_shift[i], expected_a[i] << expected_b[i]);
    CHECK_VALUE(log, l_shift[i], expected_a[i] << expected_b[i], i);

    // operator &
    CHECK_TYPE(log, and1.get(i), expected_a[i] & expected_b[i]);
    CHECK_VALUE(log, and1.get(i), expected_a[i] & expected_b[i], i);
    CHECK_TYPE(log, and1[i], expected_a[i] & expected_b[i]);
    CHECK_VALUE(log, and1[i], expected_a[i] & expected_b[i], i);

    // operator |
    CHECK_TYPE(log, or1.get(i), expected_a[i] | expected_b[i]);
    CHECK_VALUE(log, or1.get(i), expected_a[i] | expected_b[i], i);
    CHECK_TYPE(log, or1[i], expected_a[i] | expected_b[i]);
    CHECK_VALUE(log, or1[i], expected_a[i] | expected_b[i], i);

    // operator ^
    CHECK_TYPE(log, xor1.get(i), expected_a[i] ^ expected_b[i]);
    CHECK_VALUE(log, xor1.get(i), expected_a[i] ^ expected_b[i], i);
    CHECK_TYPE(log, xor1[i], expected_a[i] ^ expected_b[i]);
    CHECK_VALUE(log, xor1[i], expected_a[i] ^ expected_b[i], i);

    // operator && (range, range)
    CHECK_TYPE(log, and_log.get(i),
               test_AND<dims>(expected_a[i], expected_b[i]));
    CHECK_VALUE(log, and_log.get(i),
                test_AND<dims>(expected_a[i], expected_b[i]), i);
    CHECK_TYPE(log, and_log[i], test_AND<dims>(expected_a[i], expected_b[i]));
    CHECK_VALUE(log, and_log[i], test_AND<dims>(expected_a[i], expected_b[i]),
                i);

    // operator || (range, range)
    CHECK_TYPE(log, or_log.get(i), test_OR<dims>(expected_a[i], expected_b[i]));
    CHECK_VALUE(log, or_log.get(i), test_OR<dims>(expected_a[i], expected_b[i]),
                i);
    CHECK_TYPE(log, or_log[i], test_OR<dims>(expected_a[i], expected_b[i]));
    CHECK_VALUE(log, or_log[i], test_OR<dims>(expected_a[i], expected_b[i]), i);

    // operator + (range, size_t)
    CHECK_TYPE(log, add_size_t_r.get(i), expected_a[i] + a);
    CHECK_VALUE(log, add_size_t_r.get(i), expected_a[i] + a, i);
    CHECK_TYPE(log, add_size_t_r[i], expected_a[i] + a);
    CHECK_VALUE(log, add_size_t_r[i], expected_a[i] + a, i);

    // operator + (size_t, range)
    CHECK_TYPE(log, add_size_t_l.get(i), a + expected_a[i]);
    CHECK_VALUE(log, add_size_t_l.get(i), a + expected_a[i], i);
    CHECK_TYPE(log, add_size_t_l[i], a + expected_a[i]);
    CHECK_VALUE(log, add_size_t_l[i], a + expected_a[i], i);

    // operator - (range, size_t)
    CHECK_TYPE(log, subtract_size_t_r.get(i), expected_a[i] - a);
    CHECK_VALUE(log, subtract_size_t_r.get(i), expected_a[i] - a, i);
    CHECK_TYPE(log, subtract_size_t_r[i], expected_a[i] - a);
    CHECK_VALUE(log, subtract_size_t_r[i], expected_a[i] - a, i);

    // operator - (size_t, range)
    CHECK_TYPE(log, subtract_size_t_l.get(i), a - expected_a[i]);
    CHECK_VALUE(log, subtract_size_t_l.get(i), a - expected_a[i], i);
    CHECK_TYPE(log, subtract_size_t_l[i], a - expected_a[i]);
    CHECK_VALUE(log, subtract_size_t_l[i], a - expected_a[i], i);

    // operator / (range, size_t)
    CHECK_TYPE(log, divide_size_t_r.get(i), expected_a[i] / a);
    CHECK_VALUE(log, divide_size_t_r.get(i), expected_a[i] / a, i);
    CHECK_TYPE(log, divide_size_t_r[i], expected_a[i] / a);
    CHECK_VALUE(log, divide_size_t_r[i], expected_a[i] / a, i);

    // operator / (size_t, range)
    CHECK_TYPE(log, divide_size_t_l.get(i), a / expected_a[i]);
    CHECK_VALUE(log, divide_size_t_l.get(i), a / expected_a[i], i);
    CHECK_TYPE(log, divide_size_t_l[i], a / expected_a[i]);
    CHECK_VALUE(log, divide_size_t_l[i], a / expected_a[i], i);

    // operator * (range, size_t)
    CHECK_TYPE(log, multiply_size_t_r.get(i), expected_a[i] * a);
    CHECK_VALUE(log, multiply_size_t_r.get(i), expected_a[i] * a, i);
    CHECK_TYPE(log, multiply_size_t_r[i], expected_a[i] * a);
    CHECK_VALUE(log, multiply_size_t_r[i], expected_a[i] * a, i);

    // operator * (size_t, range)
    CHECK_TYPE(log, multiply_size_t_l.get(i), a * expected_a[i]);
    CHECK_VALUE(log, multiply_size_t_l.get(i), a * expected_a[i], i);
    CHECK_TYPE(log, multiply_size_t_l[i], a * expected_a[i]);
    CHECK_VALUE(log, multiply_size_t_l[i], a * expected_a[i], i);

    // operator % (range, size_t)
    CHECK_TYPE(log, mod_size_t_r.get(i), expected_a[i] % a);
    CHECK_VALUE(log, mod_size_t_r.get(i), expected_a[i] % a, i);
    CHECK_TYPE(log, mod_size_t_r[i], expected_a[i] % a);
    CHECK_VALUE(log, mod_size_t_r[i], expected_a[i] % a, i);

    // operator % (size_t, range)
    CHECK_TYPE(log, mod_size_t_l.get(i), a % expected_a[i]);
    CHECK_VALUE(log, mod_size_t_l.get(i), a % expected_a[i], i);
    CHECK_TYPE(log, mod_size_t_l[i], a % expected_a[i]);
    CHECK_VALUE(log, mod_size_t_l[i], a % expected_a[i], i);

    // operator << (range, size_t)
    CHECK_TYPE(log, l_shift_size_t_r.get(i), expected_a[i] << a);
    CHECK_VALUE(log, l_shift_size_t_r.get(i), expected_a[i] << a, i);
    CHECK_TYPE(log, l_shift_size_t_r[i], expected_a[i] << a);
    CHECK_VALUE(log, l_shift_size_t_r[i], expected_a[i] << a, i);

    // operator << (size_t, range)
    CHECK_TYPE(log, l_shift_size_t_l.get(i), a << expected_a[i]);
    CHECK_VALUE(log, l_shift_size_t_l.get(i), a << expected_a[i], i);
    CHECK_TYPE(log, l_shift_size_t_l[i], a << expected_a[i]);
    CHECK_VALUE(log, l_shift_size_t_l[i], a << expected_a[i], i);

    // operator >> (range, size_t)
    CHECK_TYPE(log, r_shift_size_t_r.get(i), expected_a[i] >> a);
    CHECK_VALUE(log, r_shift_size_t_r.get(i), expected_a[i] >> a, i);
    CHECK_TYPE(log, r_shift_size_t_r[i], expected_a[i] >> a);
    CHECK_VALUE(log, r_shift_size_t_r[i], expected_a[i] >> a, i);

    // operator >> (size_t, range)
    CHECK_TYPE(log, r_shift_size_t_l.get(i), a >> expected_a[i]);
    CHECK_VALUE(log, r_shift_size_t_l.get(i), a >> expected_a[i], i);
    CHECK_TYPE(log, r_shift_size_t_l[i], a >> expected_a[i]);
    CHECK_VALUE(log, r_shift_size_t_l[i], a >> expected_a[i], i);
  }

  cl::sycl::range<dims> add2(r_one);
  cl::sycl::range<dims> subtract2(r_one);
  cl::sycl::range<dims> divide2(r_one);
  cl::sycl::range<dims> multiply2(r_one);
  cl::sycl::range<dims> mod2(r_one);
  cl::sycl::range<dims> r_shift2(r_one);
  cl::sycl::range<dims> l_shift2(r_one);
  cl::sycl::range<dims> and2(r_one);
  cl::sycl::range<dims> or2(r_one);
  cl::sycl::range<dims> xor2(r_one);

  add2 += r_two;
  subtract2 -= r_two;
  divide2 /= r_two;
  multiply2 *= r_two;
  mod2 %= r_two;
  r_shift2 >>= r_two;
  l_shift2 <<= r_two;
  and2 &= r_two;
  or2 |= r_two;
  xor2 ^= r_two;

  for (i = 0; i < dims; i++) {
    // operator +=
    CHECK_TYPE(log, add2.get(i), expected_a[i] + expected_b[i]);
    CHECK_VALUE(log, add2.get(i), expected_a[i] + expected_b[i], i);
    CHECK_TYPE(log, add2[i], expected_a[i] + expected_b[i]);
    CHECK_VALUE(log, add2[i], expected_a[i] + expected_b[i], i);

    // operator -=
    CHECK_TYPE(log, subtract2.get(i), expected_a[i] - expected_b[i]);
    CHECK_VALUE(log, subtract2.get(i), expected_a[i] - expected_b[i], i);
    CHECK_TYPE(log, subtract2[i], expected_a[i] - expected_b[i]);
    CHECK_VALUE(log, subtract2[i], expected_a[i] - expected_b[i], i);

    // operator /=
    CHECK_TYPE(log, divide2.get(i), expected_a[i] / expected_b[i]);
    CHECK_VALUE(log, divide2.get(i), expected_a[i] / expected_b[i], i);
    CHECK_TYPE(log, divide2[i], expected_a[i] / expected_b[i]);
    CHECK_VALUE(log, divide2[i], expected_a[i] / expected_b[i], i);

    // operator *=
    CHECK_TYPE(log, multiply2.get(i), expected_a[i] * expected_b[i]);
    CHECK_VALUE(log, multiply2.get(i), expected_a[i] * expected_b[i], i);
    CHECK_TYPE(log, multiply2[i], expected_a[i] * expected_b[i]);
    CHECK_VALUE(log, multiply2[i], expected_a[i] * expected_b[i], i);

    // operator %=
    CHECK_TYPE(log, mod2.get(i), expected_a[i] % expected_b[i]);
    CHECK_VALUE(log, mod2.get(i), expected_a[i] % expected_b[i], i);
    CHECK_TYPE(log, mod2[i], expected_a[i] % expected_b[i]);
    CHECK_VALUE(log, mod2[i], expected_a[i] % expected_b[i], i);

    // operator >>=
    CHECK_TYPE(log, r_shift2.get(i), expected_a[i] >> expected_b[i]);
    CHECK_VALUE(log, r_shift2.get(i), expected_a[i] >> expected_b[i], i);
    CHECK_TYPE(log, r_shift2[i], expected_a[i] >> expected_b[i]);
    CHECK_VALUE(log, r_shift2[i], expected_a[i] >> expected_b[i], i);

    // operator <<=
    CHECK_TYPE(log, l_shift2.get(i), expected_a[i] << expected_b[i]);
    CHECK_VALUE(log, l_shift2.get(i), expected_a[i] << expected_b[i], i);
    CHECK_TYPE(log, l_shift2[i], expected_a[i] << expected_b[i]);
    CHECK_VALUE(log, l_shift2[i], expected_a[i] << expected_b[i], i);

    // operator &=
    CHECK_TYPE(log, and2.get(i), expected_a[i] & expected_b[i]);
    CHECK_VALUE(log, and2.get(i), expected_a[i] & expected_b[i], i);
    CHECK_TYPE(log, and2[i], expected_a[i] & expected_b[i]);
    CHECK_VALUE(log, and2[i], expected_a[i] & expected_b[i], i);

    // operator |=
    CHECK_TYPE(log, or2.get(i), expected_a[i] | expected_b[i]);
    CHECK_VALUE(log, or2.get(i), expected_a[i] | expected_b[i], i);
    CHECK_TYPE(log, or2[i], expected_a[i] | expected_b[i]);
    CHECK_VALUE(log, or2[i], expected_a[i] | expected_b[i], i);

    // operator ^=
    CHECK_TYPE(log, xor2.get(i), expected_a[i] ^ expected_b[i]);
    CHECK_VALUE(log, xor2.get(i), expected_a[i] ^ expected_b[i], i);
    CHECK_TYPE(log, xor2[i], expected_a[i] ^ expected_b[i]);
    CHECK_VALUE(log, xor2[i], expected_a[i] ^ expected_b[i], i);
  }

  bool same = (r_one == r_one);
  bool not_same = (r_one != r_two);
  bool greater = (r_one > r_two);
  bool less = (r_two < r_one);
  bool greater_eq_ne = (r_one >= r_two);
  bool less_eq_ne = (r_two <= r_one);
  bool greater_eq = (r_one >= r_one);
  bool less_eq = (r_one <= r_one);

  CHECK_VALUE(log, same, true, dims);
  CHECK_VALUE(log, not_same, true, dims);
  CHECK_VALUE(log, greater, true, dims);
  CHECK_VALUE(log, less, true, dims);
  CHECK_VALUE(log, greater_eq_ne, true, dims);
  CHECK_VALUE(log, less_eq_ne, true, dims);
  CHECK_VALUE(log, greater_eq, true, dims);
  CHECK_VALUE(log, less_eq, true, dims);
}

/** test cl::sycl::range::get(int index) return size_t
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   *  @param info, test_base::info structure as output
   */
  virtual void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   *  @param log, test transcript logging class
   */
  virtual void run(util::logger &log) override {
    try {
      /** create ranges
       */
      cl::sycl::range<1> range_1d(expected_a[0]);
      cl::sycl::range<2> range_2d(expected_a[0], expected_a[1]);
      cl::sycl::range<3> range_3d(expected_a[0], expected_a[1], expected_a[2]);

      cl::sycl::range<1> range_1d_two(expected_b[0]);
      cl::sycl::range<2> range_2d_two(expected_b[0], expected_b[1]);
      cl::sycl::range<3> range_3d_two(expected_b[0], expected_b[1],
                                      expected_b[2]);

      /** testing gets
       */
      test_gets(log, range_1d);
      test_gets(log, range_2d);
      test_gets(log, range_3d);

      /** testing operators
       */
      test_operators(log, range_1d, range_1d_two);
      test_operators(log, range_2d, range_2d_two);
      test_operators(log, range_3d, range_3d_two);

      /** testing size()
       */
      {
        CHECK_TYPE(log, range_1d.size(), expected_a[0]);
        CHECK_VALUE(log, range_1d.size(), expected_a[0], 0);

        CHECK_TYPE(log, range_2d.size(), expected_a[0] * expected_a[1]);
        CHECK_VALUE(log, range_2d.size(), expected_a[0] * expected_a[1], 1);

        CHECK_TYPE(log, range_3d.size(),
                   expected_a[0] * expected_a[1] * expected_a[2]);
        CHECK_VALUE(log, range_3d.size(),
                    expected_a[0] * expected_a[1] * expected_a[2], 2);
      }
    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace range_api__ */

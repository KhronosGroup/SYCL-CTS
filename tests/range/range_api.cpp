/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME range_api

namespace range_api__ {
using namespace sycl_cts;

static const size_t expected[] = {16, 128, 256};
static const size_t expected_two[] = {2, 16, 128};

template <int dims>
void test_gets(util::logger &log, cl::sycl::range<dims> r) {
  int i;
  for (i = 0; i < dims; i++) {
    CHECK_TYPE(log, r.get(i), expected[i]);
    CHECK_VALUE(log, r.get(i), expected[i], i);
    CHECK_TYPE(log, r[i], expected[i]);
    CHECK_VALUE(log, r[i], expected[i], i);
  }
}

template <int dims>
void test_operators(util::logger &log, cl::sycl::range<dims> r_one,
                    cl::sycl::range<dims> r_two) {
  // operators performed
  cl::sycl::range<dims> add = r_one + r_two;
  cl::sycl::range<dims> subtrack = r_one - r_two;
  cl::sycl::range<dims> divide = r_one / r_two;
  cl::sycl::range<dims> multiply = r_one * r_two;

  int i;
  for (i = 0; i < dims; i++) {
    // operator +
    CHECK_TYPE(log, add.get(i), expected[i] + expected_two[i]);
    CHECK_VALUE(log, add.get(i), expected[i] + expected_two[i], i);
    CHECK_TYPE(log, add[i], expected[i] + expected_two[i]);
    CHECK_VALUE(log, add[i], expected[i] + expected_two[i], i);

    // operator -
    CHECK_TYPE(log, subtrack.get(i), expected[i] - expected_two[i]);
    CHECK_VALUE(log, subtrack.get(i), expected[i] - expected_two[i], i);
    CHECK_TYPE(log, subtrack[i], expected[i] - expected_two[i]);
    CHECK_VALUE(log, subtrack[i], expected[i] - expected_two[i], i);

    // operator /
    CHECK_TYPE(log, divide.get(i), expected[i] / expected_two[i]);
    CHECK_VALUE(log, divide.get(i), expected[i] / expected_two[i], i);
    CHECK_TYPE(log, divide[i], expected[i] / expected_two[i]);
    CHECK_VALUE(log, divide[i], expected[i] / expected_two[i], i);

    // operator *
    CHECK_TYPE(log, multiply.get(i), expected[i] * expected_two[i]);
    CHECK_VALUE(log, multiply.get(i), expected[i] * expected_two[i], i);
    CHECK_TYPE(log, multiply[i], expected[i] * expected_two[i]);
    CHECK_VALUE(log, multiply[i], expected[i] * expected_two[i], i);
  }
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
      cl::sycl::range<1> range_1d(expected[0]);
      cl::sycl::range<2> range_2d(expected[0], expected[1]);
      cl::sycl::range<3> range_3d(expected[0], expected[1], expected[2]);

      cl::sycl::range<1> range_1d_two(expected_two[0]);
      cl::sycl::range<2> range_2d_two(expected_two[0], expected_two[1]);
      cl::sycl::range<3> range_3d_two(expected_two[0], expected_two[1],
                                      expected_two[2]);

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
    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      FAIL(log, "sycl exception caught");
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace range_api__ */

/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME range_constructors

namespace range_constructors__ {
using namespace sycl_cts;

/** test cl::sycl::range initialization
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   *  @param info, test_base::info structure as output
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   *  @param log, test transcript logging class
   */
  void run(util::logger &log) override {
    try {
      // use across all the dimensions
      size_t sizes[] = {16, 8, 4};

      // construct from a range, explicit dimensions perform deep copy and a
      // move

      // dim 1
      {
        cl::sycl::range<1> range_explicit(sizes[0]);
        if ((range_explicit[0] != sizes[0]) ||
            (range_explicit.get(0) != sizes[0])) {
          FAIL(log,
               "range with size_t was not constructed correctly for dim = 1");
        }

        cl::sycl::range<1> range_deep(
            const_cast<const cl::sycl::range<1> &>(range_explicit));
        if ((range_deep[0] != sizes[0]) || (range_deep.get(0) != sizes[0])) {
          FAIL(log,
               "range with range was not constructed correctly for dim = 1");
        }

        cl::sycl::range<1> range_moved_constr(std::move(range_explicit));
        if ((range_moved_constr[0] != sizes[0]) ||
            (range_moved_constr.get(0) != sizes[0])) {
          FAIL(log,
               "range with range was not move constructed correctly for dim = "
               "1");
        }

        cl::sycl::range<1> range_move_assign{0};
        range_move_assign = std::move(range_deep);
        if ((range_move_assign[0] != sizes[0]) ||
            (range_move_assign.get(0) != sizes[0])) {
          FAIL(log,
               "range with range was not move assigned correctly for dim = 1");
        }

        check_equality_comparable_generic(log, range_explicit,
                                          std::string("range"));
      }

      // dim 2
      {
        cl::sycl::range<2> range_explicit(sizes[0], sizes[1]);
        if ((range_explicit[0] != sizes[0]) ||
            (range_explicit.get(0) != sizes[0]) ||
            (range_explicit[1] != sizes[1]) ||
            (range_explicit.get(1) != sizes[1])) {
          FAIL(log,
               "range with size_t was not constructed correctly for dim = 2");
        }

        cl::sycl::range<2> range_deep(
            const_cast<const cl::sycl::range<2> &>(range_explicit));
        if ((range_deep[0] != sizes[0]) || (range_deep.get(0) != sizes[0]) ||
            (range_deep[1] != sizes[1]) || (range_deep.get(1) != sizes[1])) {
          FAIL(log,
               "range with range was not constructed correctly for dim = 2");
        }

        cl::sycl::range<2> range_moved_constr(std::move(range_explicit));
        if ((range_moved_constr[0] != sizes[0]) ||
            (range_moved_constr.get(0) != sizes[0]) ||
            (range_moved_constr[1] != sizes[1]) ||
            (range_moved_constr.get(1) != sizes[1])) {
          FAIL(log,
               "range with range was not move constructed correctly for dim = "
               "2");
        }

        cl::sycl::range<2> range_move_assign{0, 0};
        range_move_assign = std::move(range_deep);
        if ((range_move_assign[0] != sizes[0]) ||
            (range_move_assign.get(0) != sizes[0]) ||
            (range_move_assign[1] != sizes[1]) ||
            (range_move_assign.get(1) != sizes[1])) {
          FAIL(log,
               "range with range was not move assigned correctly for dim = 2");
        }

        check_equality_comparable_generic(log, range_explicit,
                                          std::string("range"));
      }

      // dim 3
      {
        cl::sycl::range<3> range_explicit(sizes[0], sizes[1], sizes[2]);
        if ((range_explicit[0] != sizes[0]) ||
            (range_explicit.get(0) != sizes[0]) ||
            (range_explicit[1] != sizes[1]) ||
            (range_explicit.get(1) != sizes[1]) ||
            (range_explicit[2] != sizes[2]) ||
            (range_explicit.get(2) != sizes[2])) {
          FAIL(log,
               "range with size_t was not constructed correctly for dim = 3");
        }

        cl::sycl::range<3> range_deep(
            const_cast<const cl::sycl::range<3> &>(range_explicit));
        if ((range_deep[0] != sizes[0]) || (range_deep.get(0) != sizes[0]) ||
            (range_deep[1] != sizes[1]) || (range_deep.get(1) != sizes[1]) ||
            (range_deep[2] != sizes[2]) || (range_deep.get(2) != sizes[2])) {
          FAIL(log,
               "range with range was not constructed correctly for dim = 3");
        }

        cl::sycl::range<3> range_moved_constr(std::move(range_explicit));
        if ((range_moved_constr[0] != sizes[0]) ||
            (range_moved_constr.get(0) != sizes[0]) ||
            (range_moved_constr[1] != sizes[1]) ||
            (range_moved_constr.get(1) != sizes[1]) ||
            (range_moved_constr[2] != sizes[2]) ||
            (range_moved_constr.get(2) != sizes[2])) {
          FAIL(log,
               "range with range was not move constructed correctly for dim = "
               "3");
        }

        cl::sycl::range<3> range_move_assign{0, 0, 0};
        range_move_assign = std::move(range_deep);
        if ((range_move_assign[0] != sizes[0]) ||
            (range_move_assign.get(0) != sizes[0]) ||
            (range_move_assign[1] != sizes[1]) ||
            (range_move_assign.get(1) != sizes[1]) ||
            (range_move_assign[2] != sizes[2]) ||
            (range_move_assign.get(2) != sizes[2])) {
          FAIL(log,
               "range with range was not move assigned correctly for dim = 3");
        }

        check_equality_comparable_generic(log, range_explicit,
                                          std::string("range"));
      }
    } catch (const cl::sycl::exception &e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace range_constructors__ */

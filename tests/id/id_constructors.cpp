/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME id_constructors

namespace TEST_NAMESPACE {
using namespace sycl_cts;

/** test cl::sycl::id initialization
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  virtual void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  virtual void run(util::logger &log) override {
    try {
      // use across all the dimensions
      size_t sizes[] = {16, 8, 4};

      // construct from a range and perform deep copy

      // dim 1
      {
        cl::sycl::range<1> range(sizes[0]);
        cl::sycl::id<1> id(range);
        cl::sycl::id<1> id_deep(id);
      }

      // dim 2
      {
        cl::sycl::range<2> range(sizes[0], sizes[1]);
        cl::sycl::id<2> id(range);
        cl::sycl::id<2> id_deep(id);
      }

      // dim 3
      {
        cl::sycl::range<3> range(sizes[0], sizes[1], sizes[2]);
        cl::sycl::id<3> id(range);
        cl::sycl::id<3> id_deep(id);
      }

      // construct from an item

      // dim 1
      {
        auto q = util::get_cts_object::queue();
        q.submit([&](cl::sycl::handler &cgh) {
          auto my_range = cl::sycl::range<1>(sizes[0]);

          auto my_kernel = [=](cl::sycl::item<1> item) {
            cl::sycl::id<1> id(item);
          };
          cgh.parallel_for<class _it1>(my_range, my_kernel);
        });

        q.wait_and_throw();
      }

      // dim 2
      {
        auto q = util::get_cts_object::queue();
        q.submit([&](cl::sycl::handler &cgh) {
          auto my_range = cl::sycl::range<2>(sizes[0], sizes[1]);

          auto my_kernel = [=](cl::sycl::item<2> item) {
            cl::sycl::id<2> id(item);
          };
          cgh.parallel_for<class _it2>(my_range, my_kernel);
        });

        q.wait_and_throw();
      }

      // dim 3
      {
        auto q = util::get_cts_object::queue();
        q.submit([&](cl::sycl::handler &cgh) {
          auto my_range = cl::sycl::range<3>(sizes[0], sizes[1], sizes[2]);

          auto my_kernel = [=](cl::sycl::item<3> item) {
            cl::sycl::id<3> id(item);
          };
          cgh.parallel_for<class _it3>(my_range, my_kernel);
        });

        q.wait_and_throw();
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

} /* namespace id_constructors__ */

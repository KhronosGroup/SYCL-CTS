/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME asynchronous_exceptions

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace cl::sycl;

/**
 */
class TEST_NAME : public util::test_base {
 public:
  struct async_handler_functor {
    vector_class<exception_ptr_class> excps;
    void operator()(cl::sycl::exception_list l) {
      for (auto &e : l) {
        excps.push_back(e);
      }
    }
  };

  static void async_handler_function(cl::sycl::exception_list l) {
    /*no access to logger at this point*/
    for (auto &e : l) {
      throw e;
    }
  }

  /** log the exceptions.
   */
  void check_exceptions(util::logger &log,
                        vector_class<exception_ptr_class> &excps) const {
    for (auto &e : excps) {
      try {
        throw e;
      } catch (const exception &e) {
        // Check methods
        cl::sycl::string_class sc = e.what();
        if (e.has_context()) {
          context c = e.get_context();
        }
        cl::sycl::cl_int ci = e.get_cl_code();

        log_exception(log, e);
        FAIL(log, "An exception should not really have been thrown");
      }
    }
  }

  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  void run(util::logger &log) override {
    /*test lambda async handler*/
    {
      vector_class<exception_ptr_class> excps;
      cl::sycl::function_class<void(cl::sycl::exception_list)>
          asyncHandlerLambda = [&excps](cl::sycl::exception_list l) {
            for (auto &e : l) {
              excps.push_back(e);
            }
          };

      cts_selector selector;
      cl::sycl::queue q(selector, asyncHandlerLambda);

      q.submit(
          [&](handler &cgh) { cgh.single_task<class TEST_NAME>([=]() {}); });

      // Should not throw exceptions
      q.wait();

      check_exceptions(log, excps);
    }

    /*test functor async handler*/
    {
      async_handler_functor asyncHandlerFunctor;

      cts_selector selector;
      cl::sycl::queue q(selector, asyncHandlerFunctor);

      q.submit(
          [&](handler &cgh) { cgh.single_task<class TEST_NAME_2>([=]() {}); });

      // Should not throw exceptions
      q.wait();

      check_exceptions(log, asyncHandlerFunctor.excps);
    }

    /*test function async handler*/
    {
      cts_selector selector;
      cl::sycl::queue q(selector, async_handler_function);

      q.submit(
          [&](handler &cgh) { cgh.single_task<class TEST_NAME_3>([=]() {}); });

      // Should not throw exceptions
      q.wait();
    }
  }
};

util::test_proxy<TEST_NAME> proxy;

}  // TEST_NAMESPACE

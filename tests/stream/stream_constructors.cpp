/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME stream_constructors

namespace stream_constructors__ {
using namespace sycl_cts;

/** tests the constructors for cl::sycl::stream
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
      /** check default constructor and destructor
      */
      { cl::sycl::stream os; }

      /** check (size_t, size_t) constructor
      */
      {
        size_t bufferSize = 2048;
        size_t maxStatementSize = 80;
        cl::sycl::stream os(bufferSize, maxStatementSize);

        auto size = os.get_size();

        if (size != 2048) {
          FAIL(log,
               "cl::sycl::context::get_size() returned an incorrect value.");
        }

        maxStatementSize = os.get_max_statement_size();

        if (maxStatementSize != 80) {
          FAIL(log,
               "cl::sycl::context::get_max_statement_size() returned an "
               "incorrect value.");
        }
      }

      /** check copy constructor
      */
      {
        cl::sycl::stream osA;
        cl::sycl::stream osB(osA);
      }

      /** check assignment operator
      */
      {
        cl::sycl::stream osA;
        cl::sycl::stream osB = osA;
      }
    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      FAIL(log, "a sycl exception was caught");
    }
  }
};

// register this test with the test_collection.
util::test_proxy<TEST_NAME> proxy;

} /* namespace stream_constructors__ */

/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "executor.h"
#include "singleton.h"
#include "printer.h"
#include "collection.h"
#include "logger.h"

namespace sycl_cts {
namespace util {

/** execute all tests in the collection
 */
bool executor::run_all() {
  // find the number of tests in the collection
  const int32_t nTests = get<collection>().get_test_count();
  int32_t numPassed = 0;

  // iterate over all tests
  for (int32_t i = 0; i < nTests; i++) {
    // locate a specific test
    collection::test_info &info = get<collection>().get_test(i);
    test_base *test = info.m_test;

    // do not execute any test marked to be skipped
    if (info.m_skip) {
      continue;
    }

    // scope for the logger
    {
      // log for this test execution
      logger logger;

      // write the test info header
      test_base::info testInfo;
      test->get_info(testInfo);
      logger.preamble(testInfo);

      logger.test_start();

      // we must install an exception handler here so that if a test
      // fails to catch a thrown exception it wont crash the entire
      // test suite
      try {
        // ask the test to set itself up
        if (test->setup(logger)) {
          // ask the test to execute
          test->run(logger);
        }
        // enforce that each test must give a result
        assert(logger.get_result() != logger::epending);

        if ((logger.get_result() == logger::epass) ||
            (logger.get_result() == logger::eskip)) {
          ++numPassed;
        }

        // ask the test to clean up after itself
        test->cleanup();
      } catch (...) {
        logger.fail("Exception thrown and not caught by test case!", 0);
      }

      logger.test_end();

      // if we received a fatal error then we must exit
      if (logger.get_result() == logger::efatal) {
        break;
      }
    }
  }
  {
    logger log;
    auto successRate =
        100 * static_cast<float>(numPassed) / static_cast<float>(nTests);
    if ((successRate > 99.f) && (numPassed < nTests)) {
      successRate = 99.f;
    }
    log.note("Passed %d/%d tests (%.0f%%)", numPassed, nTests, successRate);
  }
  return (numPassed == nTests);
}

}  // namespace util
}  // namespace sycl_cts

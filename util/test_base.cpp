/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provide implementation of test_base class functions
//
*******************************************************************************/

#include "test_base.h"

#include <sycl/sycl.hpp>

#include "../tests/common/macros.h"
#include "logger.h"
#include "sycl_exceptions.h"

// conformance test suite namespace
namespace sycl_cts {
namespace util {

void test_base::run_test(class logger &log) {
  try {
    if (this->setup(log))
      this->run(log);
  } catch (const sycl::exception &e) {
    log_exception(log, e);
    auto errorMsg = "a SYCL exception was caught: " + std::string(e.what());
    FAIL(log, errorMsg);
  } catch (const std::exception &e) {
    auto errorMsg = "an exception was caught: " + std::string(e.what());
    FAIL(log, errorMsg);
  }
}

}  // namespace util
}  // namespace sycl_cts

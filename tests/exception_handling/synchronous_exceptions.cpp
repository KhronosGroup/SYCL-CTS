/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#include <limits>

#define TEST_NAME synchronous_exceptions

namespace TEST_NAMESPACE {
using namespace sycl_cts;

class custom_exception : public sycl::exception {
 public:
  struct reference {
    static std::string what() {
      return "custom_exception";
    }
    static std::error_code code() {
      return sycl::make_error_code(sycl::errc::runtime);
    }
    static const std::error_category& category() {
      return sycl::sycl_category();
    }
  };
  custom_exception(sycl::context ctx) :
      sycl::exception(ctx, reference::code(), reference::what()) {}
};

/** @brief Verify exception propagation from command group
 */
void check_exception_api(util::logger &log) {
  auto q = util::get_cts_object::queue();
  auto ctx = q.get_context();

  try {
    q.submit([&](cl::sycl::handler &cgh) {
      cgh.single_task<class TEST_NAME>([=]() {});

      throw custom_exception(ctx);
    });
    q.wait_and_throw();
  } catch (const custom_exception &e) {
    log_exception(log, e);

    if (e.what() != custom_exception::reference::what()) {
      FAIL(log, "invalid value for what()");
    }
    if (e.code() != custom_exception::reference::code()) {
      FAIL(log, "invalid value for code()");
    }
    if (e.category() != custom_exception::reference::category()) {
      FAIL(log, "invalid value for category()");
    }
    if (!e.has_context()) {
      FAIL(log, "invalid value for has_context()");
    } else if (e.get_context() != ctx) {
      FAIL(log, "invalid value for get_context()");
    }
    return;
  }
  FAIL(log, "No expected exception was thrown");
}

/** @brief Verify exception usage for synchronous errors
 */
void check_exception_usage(util::logger &log) {
  try {
    auto device = util::get_cts_object::device();
    const auto partition = info::partition_property::partition_equally;
    const size_t count = std::numeric_limits<size_t>::max();

    // Either errc::feature_not_supported or errc::invalid should be thrown
    device.create_sub_devices<partition>(count);

  } catch (const sycl::exception &e) {
    const auto code = e.code();

    if ((code == sycl::errc::feature_not_supported) ||
        (code == sycl::errc::invalid)) {
      log_exception(log, e);
      return;
    }
    // Unexpected exception; re-trow
    throw;
  }
  FAIL(log, "No expected exception was thrown");
}

/**
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  void run(util::logger &log) override {
    try {
      check_exception_api(log);
      check_exception_usage(log);
    } catch (const cl::sycl::exception &e) {
      log_exception(log, e);
      auto errorMsg = std::string("a SYCL exception was caught: ") + e.what();
      FAIL(log, errorMsg);
    }
  }
};

util::test_proxy<TEST_NAME> proxy;

}  // TEST_NAMESPACE

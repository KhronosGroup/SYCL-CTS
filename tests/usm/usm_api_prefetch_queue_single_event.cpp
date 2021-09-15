/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for overload of the queue::prefetch() method with 1 event
//
*******************************************************************************/

#include "../common/common.h"
#include "usm_api.h"

#define TEST_NAME usm_api_prefetch_queue_single_event

namespace TEST_NAMESPACE {
using namespace usm_api;

template <typename T>
using run_tests = run_all_tests<T, tests::prefetch, caller::queue, 1_events>;

/** Test instance
 */
class TEST_NAME : public sycl_cts::util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  void run(sycl_cts::util::logger &log) override {
    try {
      run_tests<int>{}(log, "int");
    } catch (const sycl::exception &ex) {
      log_exception(log, ex);
      auto errorMsg = "a SYCL exception was caught: " + std::string(ex.what());
      FAIL(log, errorMsg);
    } catch (const std::exception &ex) {
      auto errorMsg = "an exception was caught: " + std::string(ex.what());
      FAIL(log, errorMsg);
    }
  }
};

// construction of this proxy will register the above test
sycl_cts::util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE

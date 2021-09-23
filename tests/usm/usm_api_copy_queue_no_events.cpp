/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for overload of the queue::copy() method with no events
//
*******************************************************************************/

#include "../common/common.h"
#include "usm_api.h"

#define TEST_NAME usm_api_copy_queue_no_events

namespace TEST_NAMESPACE {
using namespace usm_api;

template <typename T, template <typename, size_t> class checkT>
using run_tests = run_all_tests<T, checkT, caller::queue, 0_events>;

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
    run_tests<int, tests::copy_from_non_usm>{}(log);
    run_tests<int, tests::copy_from_host>{}(log);
    run_tests<int, tests::copy_from_shared>{}(log);
    run_tests<int, tests::copy_from_device>{}(log);
  }
};

// construction of this proxy will register the above test
sycl_cts::util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE

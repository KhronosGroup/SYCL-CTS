/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for sycl::join for zero bundles
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/get_cts_string.h"
#include "sycl_join.h"

#define TEST_NAME sycl_join_zero_kernel_bundles

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace sycl_cts::tests::sycl_join;

/** @brief Provides check for joining of zero kernel bundles of State given
 *  @tparam State sycl::bundle_state
 */
template <sycl::bundle_state State>
void run_verification(util::logger &log) {
  auto joined_kb = sycl::join<State>({});
  if (!joined_kb.empty()) {
    FAIL(log,
         "sycl::join should return empty kernel_bundle for an empty "
         "vector of kernel bundles and state " +
             get_cts_string::for_bundle_state<State>());
  }
}

/** test sycl::join for zero bundles
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
  void run(util::logger &log) override {
    // Execute tests for all sycl::bundle_states
    run_verification<sycl::bundle_state::input>(log);
    run_verification<sycl::bundle_state::object>(log);
    run_verification<sycl::bundle_state::executable>(log);
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE

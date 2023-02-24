/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for sycl::join for single bundle (once, twice)
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/get_cts_string.h"
#include "sycl_join.h"

#define TEST_NAME sycl_join_single_kernel_bundle

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace sycl_cts::tests::sycl_join;

/** @brief Provides check for joining single kernel bundle (once, twice)
 *  @tparam State sycl::bundle_state
 *  @details Get kernel bundle and perform check for two scenarios:
 *            - sycl::join({kernel_bundle}) (once)
 *            - sycl::join({kernel_bundle, kernel_bundle}) (twice)
 *           Then check that joined kernel bundle has the same set of device
 *           images as original one in both scenarios
 */
template <sycl::bundle_state State>
void run_verification(util::logger &log) {
  auto queue = util::get_cts_object::queue();
  const auto ctx = queue.get_context();
  const auto dev = queue.get_device();

  auto kb = sycl::get_kernel_bundle<State>(ctx);

  // Check joined single bundle once
  {
    auto joined_kb = sycl::join<State>({kb});
    if (!check_dev_images_equal(log, kb, joined_kb)) {
      FAIL(log,
           "Set of device images is not the same before and after joining "
           "a single kernel_bundle once with state: " +
               get_cts_string::for_bundle_state<State>());
    }
  }

  // Check joined single bundle twice
  {
    auto joined_kb = sycl::join<State>({kb, kb});
    if (!check_dev_images_equal(log, kb, joined_kb)) {
      FAIL(log,
           "Set of device images is not the same before and after joining "
           "a single kernel_bundle twice with state: " +
               get_cts_string::for_bundle_state<State>());
    }
  }
}

/** test sycl::join for single bundle (once, twice)
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

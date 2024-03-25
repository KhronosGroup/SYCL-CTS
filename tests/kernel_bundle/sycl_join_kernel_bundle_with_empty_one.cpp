/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for sycl::join with empty bundle
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/get_cts_string.h"
#include "sycl_join.h"

#define TEST_NAME sycl_join_kernel_bundle_with_empty_one

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace sycl_cts::tests::sycl_join;

/** @brief Provides check for joining kernel bundle with an empty kernel bundle
 *  @tparam State sycl::bundle_state
 *  @details Get one kernel bundle as usual and second kernel bundle with
 *           selector that always returns false to make it empty. Perform check
 *           for two scenarios:
 *            - sycl::join(kernel_bundle, empty_kernel_bundle)
 *            - sycl::join(empty_kernel_bundle, kernel_bundle)
 *           Then check that joined kernel bundle has the same set of device
 *           images as non-empty kernel bundle in both scenarios
 */
template <sycl::bundle_state State>
void run_verification(util::logger &log) {
  auto queue = util::get_cts_object::queue();
  const auto ctx = queue.get_context();

  auto kb = sycl::get_kernel_bundle<State>(ctx);

  // Selector that always returns false. Used to get empty kernel_bundle
  auto false_selector = [](const sycl::device_image<State> &) { return false; };
  auto empty_kb = sycl::get_kernel_bundle<State>(ctx, false_selector);

  // Check joined bundles in such order: (kernel_bundle, empty_kernel_bundle)
  {
    auto joined_kb = sycl::join<State>({kb, empty_kb});
    if (!check_dev_images_equal(log, kb, joined_kb)) {
      FAIL(log,
           "Set of device images is not the same before and after joining "
           "with an empty kernel_bundle in such order: "
           "(kernel_bundle, empty_kernel_bundle) for state: " +
               get_cts_string::for_bundle_state<State>());
    }
  }

  // Check joined bundles in such order: (empty_kernel_bundle, kernel_bundle)
  {
    auto joined_kb = sycl::join<State>({empty_kb, kb});
    if (!check_dev_images_equal(log, kb, joined_kb)) {
      FAIL(log,
           "Set of device images is not the same before and after joining "
           "with an empty kernel_bundle in such order: "
           "(empty_kernel_bundle, kernel_bundle) for state: " +
               get_cts_string::for_bundle_state<State>());
    }
  }
}

/** test sycl::join with empty bundle
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

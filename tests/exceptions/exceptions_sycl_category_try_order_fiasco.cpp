/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for sycl::sycl_category function with try to provoke static
//  initialization order fiasco
//
*******************************************************************************/

#include "exceptions.h"
#include "exceptions_sycl_category_common.h"

#define TEST_NAME exceptions_sycl_category_try_order_fiasco

namespace TEST_NAMESPACE {

using namespace sycl_cts;
using namespace exceptions_sycl_category_common;

/** @brief Try to provoke Static Initialization Order Fiasco
 *  @details This global static instance depends on sycl_category static
 * instance, so we have a possibility of SIOF triggering based on the linkage
 * order
 */
const static test_result_checker static_instance(sycl::sycl_category());

/** Test instance
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
    {
      static_instance.check_results(
          "while trying to provoke static initialization order fiasco", log);
    }
  }
};

util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE

/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for multi_ptr members tests with core data types.
//
*******************************************************************************/

#include "../common/common.h"
#include "multi_ptr_members.h"
#include "multi_ptr_common.h"

#define TEST_NAME multi_ptr_members_core

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace multi_ptr_members;

/** tests the api for explicit pointers
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute this test
   */
  void run(util::logger &log) override {
    const auto types = multi_ptr_common::get_types();
    const auto composite_types = multi_ptr_common::get_composite_types();
    for_all_types<run_test_with_chosen_data_type>(types, log);
    for_all_types<run_test_with_chosen_data_type>(composite_types, log);
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */

/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for specialization constants with same name and internal
//  linkage for double (2nd translation unit)
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/type_coverage.h"

// Index of TU
#define SC_SN_IL_TU_NUM 2

#include "spec_constants_same_name_inter_link.h"

#define TEST_NAME specialization_constants_same_name_inter_link_2nd_tu_fp64

namespace TEST_NAMESPACE {
using namespace sycl_cts;

/** test specialization constants for double
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
    using namespace specialization_constants_same_name_inter_link;
    using namespace get_spec_const;
    sc_run_test_fp64<SC_SN_IL_TU_NUM, sc_no_kernel_bundle>(log);
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE

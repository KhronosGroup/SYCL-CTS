/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
// Provides tests for specialization constants usage via handler for double
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/type_coverage.h"
#include "spec_constants_via_handler_common.h"

#define TEST_NAME specialization_constants_via_handler_fp64

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
    using namespace specialization_constants_via_handler_common;
    {
      auto queue = util::get_cts_object::queue();
      if (!queue.get_device().has(sycl::aspect::fp64)) {
        log.note("Device does not support double precision floating point "
                 "operations");
        return;
      }
#if !SYCL_CTS_ENABLE_FULL_CONFORMANCE
      check_spec_constant_with_handler_for_type<double> fp64_test{};
      fp64_test(log, "double");
#else
      for_type_vectors_marray<check_spec_constant_with_handler_for_type,
                              double>(log, "double");
#endif
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace spec_const__ */

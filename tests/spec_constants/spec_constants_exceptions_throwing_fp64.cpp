/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for specialization constants throwing exceptions for double
//  when expected
//
*******************************************************************************/

#include "../common/common.h"

#include "spec_constants_exceptions_throwing_common.h"

#define TEST_NAME specialization_constants_exceptions_throwing_fp64

namespace TEST_NAMESPACE {
using namespace sycl_cts;

/** test that specialization constants throws exceptions with double
    when expected
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
      auto queue = util::get_cts_object::queue();
      if (!queue.get_device().has(sycl::aspect::fp64)) {
        log.note(
            "Device does not support double precision floating point "
            "operations");
        return;
      }
#if !SYCL_CTS_ENABLE_FULL_CONFORMANCE
      check_spec_constant_exception_throw_for_type<double> fp64_test{};
      fp64_test(log, "double");
#else
      for_type_vectors_marray<check_spec_constant_exception_throw_for_type,
                              double>(log, "double");
#endif
    }
  }
};

// construction of this proxy will register the test above
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE

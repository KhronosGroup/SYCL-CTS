/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
// Provides buffer api tests for double and sycl::cl_double
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/type_coverage.h"
#include "buffer_api_common.h"

#define TEST_NAME buffer_api_fp64

namespace TEST_NAMESPACE {
using namespace sycl_cts;

/** test sycl::buffer API
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
    auto queue = util::get_cts_object::queue();
    if (!queue.get_device().has(sycl::aspect::fp64)) {
      log.note(
          "Device does not support double precision floating point operations");
      return;
    }
    for_type_and_vectors<buffer_api_common::check_buffer_api_for_type, double>(
        log, "double");
#ifdef SYCL_CTS_ENABLE_FULL_CONFORMANCE
    for_type_and_vectors<buffer_api_common::check_buffer_api_for_type,
                         sycl::cl_double>(log, "sycl::cl_double");
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE

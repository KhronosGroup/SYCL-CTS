/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/type_coverage.h"
#include "./../../util/math_helper.h"
#include "accessor_api_local_common.h"

#include <array>
#include <numeric>
#include <sstream>

#define TEST_NAME accessor_api_local_fp64

namespace TEST_NAMESPACE {

using namespace sycl_cts;
using namespace accessor_utility;

/** tests the api for cl::sycl::accessor
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
    try {
      auto queue = util::get_cts_object::queue();

      if (!queue.get_device().has_extension("cl_khr_fp64")) {
        log.note(
            "Device does not support double precision floating point "
            "operations");
        return;
      }

#ifndef SYCL_CTS_FULL_CONFORMANCE
      // Specific set of types to cover during ordinary compilation

      /** check local accessor api for double
       */
      check_local_accessor_api_type<double>()(log, queue, "double");

      /** check local accessor api for vec
       */
      check_local_accessor_api_type<cl::sycl::double3>()(log, queue, "double");
#else
      // Extended type coverage
      for_type_and_vectors<check_local_accessor_api_type, double>(
          log, queue);
      for_type_and_vectors<check_local_accessor_api_type, cl::sycl::cl_double>(
          log, queue);

#endif // SYCL_CTS_FULL_CONFORMANCE

      queue.wait_and_throw();
    } catch (const cl::sycl::exception &e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

/** register this test with the test_collection
*/
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE

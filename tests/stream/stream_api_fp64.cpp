/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides stream tests for double and cl::sycl::cl_double
//
*******************************************************************************/

#include "stream_api_common.h"

#define TEST_NAME stream_api_fp64

namespace TEST_NAMESPACE {

using namespace sycl_cts;

class test_kernel;

/** test cl::sycl::stream interface
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
    try {
      // Check stream operator for cl::sycl::cl_double and double
      auto testQueue = util::get_cts_object::queue();

      if (!testQueue.get_device().has_extension("cl_khr_fp64")) {
        log.note(
            "Device does not support double precision floating point operations");
        return;
      }

      testQueue.submit([&](cl::sycl::handler &cgh) {

        cl::sycl::stream os(2048, 80, cgh);

        cgh.single_task<class test_kernel>([=]() {
          check_all_vec_dims(os, double(5.5));
          check_all_vec_dims(os, cl::sycl::cl_double(5.5));
        });
      });

      testQueue.wait_and_throw();

    } catch (const cl::sycl::exception &e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// register this test with the test_collection.
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */

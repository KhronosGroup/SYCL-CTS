/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#define TEST_NAME accessor_constructors_image

#include "../common/common.h"
#include "accessor_constructors_utility.h"
#include "accessor_constructors_image_utility.h"

namespace TEST_NAMESPACE {

/** tests the constructors for cl::sycl::accessor
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  template <typename T>
  void check_all_dims(util::logger &log, cl::sycl::queue &queue) {
    image_accessor_dims<T, 1>::check(log, queue);
    image_accessor_dims<T, 2>::check(log, queue);
    image_accessor_dims<T, 3>::check(log, queue);
    image_array_accessor_dims<T, 1>::check(log, queue);
    image_array_accessor_dims<T, 2>::check(log, queue);
  }

  /** execute this test
   */
  void run(util::logger &log) override {
    try {
      auto queue = util::get_cts_object::queue();

      try {
        /** check accessor constructors for cl_int4
         */
        check_all_dims<cl::sycl::cl_int4>(log, queue);

        /** check accessor constructors for cl_uint4
         */
        check_all_dims<cl::sycl::cl_uint4>(log, queue);

        /** check accessor constructors for cl_float4
         */
        check_all_dims<cl::sycl::cl_float4>(log, queue);

        queue.wait_and_throw();
      } catch (const cl::sycl::feature_not_supported &fnse) {
        if (!queue.get_device()
                  .get_info<cl::sycl::info::device::image_support>()) {
          log.note("device does not support images -- skipping check");
        } else {
          throw;
        }
      }
    } catch (const cl::sycl::exception &e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE

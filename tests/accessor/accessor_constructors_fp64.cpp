/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#define TEST_NAME accessor_constructors_fp64

#include "../common/common.h"
#include "accessor_constructors_buffer_utility.h"
#include "accessor_constructors_image_utility.h"
#include "accessor_constructors_local_utility.h"

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

      /** check accessor constructors for double (fp64)
       */
      buffer_accessor_dims<
          double, 0, is_host_buffer::false_t,
          cl::sycl::access::placeholder::false_t>::check(log, queue);
      buffer_accessor_dims<
          double, 1, is_host_buffer::false_t,
          cl::sycl::access::placeholder::false_t>::check(log, queue);
      buffer_accessor_dims<
          double, 2, is_host_buffer::false_t,
          cl::sycl::access::placeholder::false_t>::check(log, queue);
      buffer_accessor_dims<
          double, 3, is_host_buffer::false_t,
          cl::sycl::access::placeholder::false_t>::check(log, queue);
      buffer_accessor_dims<
          double, 0, is_host_buffer::true_t,
          cl::sycl::access::placeholder::false_t>::check(log, queue);
      buffer_accessor_dims<
          double, 1, is_host_buffer::true_t,
          cl::sycl::access::placeholder::false_t>::check(log, queue);
      buffer_accessor_dims<
          double, 2, is_host_buffer::true_t,
          cl::sycl::access::placeholder::false_t>::check(log, queue);
      buffer_accessor_dims<
          double, 3, is_host_buffer::true_t,
          cl::sycl::access::placeholder::false_t>::check(log, queue);

      buffer_accessor_dims<double, 0, is_host_buffer::false_t,
                           cl::sycl::access::placeholder::true_t>::check(log,
                                                                         queue);
      buffer_accessor_dims<double, 1, is_host_buffer::false_t,
                           cl::sycl::access::placeholder::true_t>::check(log,
                                                                         queue);
      buffer_accessor_dims<double, 2, is_host_buffer::false_t,
                           cl::sycl::access::placeholder::true_t>::check(log,
                                                                         queue);
      buffer_accessor_dims<double, 3, is_host_buffer::false_t,
                           cl::sycl::access::placeholder::true_t>::check(log,
                                                                         queue);

      local_accessor_dims<double, 0>::check(log, queue);
      local_accessor_dims<double, 1>::check(log, queue);
      local_accessor_dims<double, 2>::check(log, queue);
      local_accessor_dims<double, 3>::check(log, queue);

      queue.wait_and_throw();
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

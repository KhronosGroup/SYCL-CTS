/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#define TEST_NAME accessor_constructors_fp16

#include "../common/common.h"
#include "accessor_constructors_buffer_utility.h"
#include "accessor_constructors_image_utility.h"
#include "accessor_constructors_local_utility.h"
#include "accessor_constructors_utility.h"

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

  void check_all_dims(util::logger &log, cl::sycl::queue &queue) {}

  template <typename T>
  void checkBufferAndLocal(util::logger &log, cl::sycl::queue &queue) {
    buffer_accessor_dims<T, 0, is_host_buffer::false_t,
                         cl::sycl::access::placeholder::false_t>::check(log,
                                                                        queue);
    buffer_accessor_dims<T, 1, is_host_buffer::false_t,
                         cl::sycl::access::placeholder::false_t>::check(log,
                                                                        queue);
    buffer_accessor_dims<T, 2, is_host_buffer::false_t,
                         cl::sycl::access::placeholder::false_t>::check(log,
                                                                        queue);
    buffer_accessor_dims<T, 3, is_host_buffer::false_t,
                         cl::sycl::access::placeholder::false_t>::check(log,
                                                                        queue);
    buffer_accessor_dims<T, 0, is_host_buffer::true_t,
                         cl::sycl::access::placeholder::false_t>::check(log,
                                                                        queue);
    buffer_accessor_dims<T, 1, is_host_buffer::true_t,
                         cl::sycl::access::placeholder::false_t>::check(log,
                                                                        queue);
    buffer_accessor_dims<T, 2, is_host_buffer::true_t,
                         cl::sycl::access::placeholder::false_t>::check(log,
                                                                        queue);
    buffer_accessor_dims<T, 3, is_host_buffer::true_t,
                         cl::sycl::access::placeholder::false_t>::check(log,
                                                                        queue);

    buffer_accessor_dims<T, 0, is_host_buffer::false_t,
                         cl::sycl::access::placeholder::true_t>::check(log,
                                                                       queue);
    buffer_accessor_dims<T, 1, is_host_buffer::false_t,
                         cl::sycl::access::placeholder::true_t>::check(log,
                                                                       queue);
    buffer_accessor_dims<T, 2, is_host_buffer::false_t,
                         cl::sycl::access::placeholder::true_t>::check(log,
                                                                       queue);
    buffer_accessor_dims<T, 3, is_host_buffer::false_t,
                         cl::sycl::access::placeholder::true_t>::check(log,
                                                                       queue);

    local_accessor_dims<T, 0>::check(log, queue);
    local_accessor_dims<T, 1>::check(log, queue);
    local_accessor_dims<T, 2>::check(log, queue);
    local_accessor_dims<T, 3>::check(log, queue);
  }

  /** execute this test
   */
  void run(util::logger &log) override {
    try {
      auto queue = util::get_cts_object::queue();

      if (!queue.get_device().has_extension("cl_khr_fp16")) {
        log.note(
            "Device does not support half precision floating point operations");
        return;
      }

      /** check accessor constructors for cl_half
       */
      checkBufferAndLocal<cl::sycl::cl_half>(log, queue);

      /** check accessor constructors for cl_half
       */
      checkBufferAndLocal<cl::sycl::cl_half2>(log, queue);

      /** check accessor constructors for cl_half
       */
      checkBufferAndLocal<cl::sycl::cl_half3>(log, queue);

      /** check accessor constructors for cl_half
       */
      checkBufferAndLocal<cl::sycl::cl_half4>(log, queue);

      /** check accessor constructors for cl_half
       */
      checkBufferAndLocal<cl::sycl::cl_half8>(log, queue);

      /** check accessor constructors for cl_half
       */
      checkBufferAndLocal<cl::sycl::cl_half16>(log, queue);

      /** check image accessor cl_half4 variants
       */
      image_accessor_dims<cl::sycl::cl_half4, 1>::check(log, queue);
      image_accessor_dims<cl::sycl::cl_half4, 2>::check(log, queue);
      image_accessor_dims<cl::sycl::cl_half4, 3>::check(log, queue);
      image_array_accessor_dims<cl::sycl::cl_half4, 1>::check(log, queue);
      image_array_accessor_dims<cl::sycl::cl_half4, 2>::check(log, queue);

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

/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#define TEST_NAME accessor_constructors_buffer_placeholder

#include "../common/common.h"
#include "accessor_constructors_utility.h"
#include "accessor_constructors_buffer_utility.h"

namespace TEST_NAMESPACE {

struct user_struct {
  float a;
  int b;
  char c;
};

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
  }

  /** execute this test
   */
  void run(util::logger &log) override {
    try {
      auto queue = util::get_cts_object::queue();

      /** check accessor constructors for int
       */
      check_all_dims<int>(log, queue);

      /** check accessor constructors for float
       */
      check_all_dims<float>(log, queue);

      /** check accessor constructors for char
       */
      check_all_dims<char>(log, queue);

      /** check accessor constructors for vec
       */
      check_all_dims<cl::sycl::int2>(log, queue);

      /** check accessor constructors for vec
       */
      check_all_dims<cl::sycl::int3>(log, queue);

      /** check accessor constructors for vec
       */
      check_all_dims<cl::sycl::float4>(log, queue);

      /** check accessor constructors for user_struct
       */
      check_all_dims<user_struct>(log, queue);

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

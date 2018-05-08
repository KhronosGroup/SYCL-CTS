/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME handler_api

namespace TEST_NAMESPACE {
using namespace sycl_cts;

struct simple_struct {
  int a;
  float b;
};

/** tests the API for cl::sycl::handler
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
      auto queue = util::get_cts_object::queue();
      const auto range = cl::sycl::range<1>(1);
      const auto offset = cl::sycl::id<1>(0);

      {
        auto buffer = cl::sycl::buffer<int, 1>(range);

        log.note("Check set_arg() methods");
        queue.submit([&](cl::sycl::handler &cgh) {
          auto accessor =
              buffer.get_access<cl::sycl::access::mode::read_write,
                                cl::sycl::access::target::global_buffer>(cgh);
          // Check set_arg(int, accessor)
          cgh.set_arg(0, accessor);

          // Check set_arg(int, int)
          {
            int scalar = 5;
            cgh.set_arg(1, scalar);
          }

          // Check set_arg(int, sampler)
          {
            cl::sycl::sampler sampler(
                cl::sycl::coordinate_normalization_mode::normalized,
                cl::sycl::addressing_mode::clamp,
                cl::sycl::filtering_mode::nearest);
            cgh.set_arg(2, sampler);
          }

          // Check set_arg(int, trivially-copyable standard layout custom type)
          {
            simple_struct custom{3, .14f};
            cgh.set_arg(3, custom);
          }

          cgh.single_task<class test_set_arg>([=]() {});
        });

        log.note("Check set_args() method");
        queue.submit([&](cl::sycl::handler &cgh) {
          auto accessor =
              buffer.get_access<cl::sycl::access::mode::read_write,
                                cl::sycl::access::target::global_buffer>(cgh);
          int scalar = 5;
          cl::sycl::sampler sampler(
              cl::sycl::coordinate_normalization_mode::normalized,
              cl::sycl::addressing_mode::clamp,
              cl::sycl::filtering_mode::nearest);
          simple_struct custom{3, .14f};

          // Check set_args(Ts...)
          cgh.set_args(accessor, scalar, sampler, custom);

          cgh.single_task<class test_set_args>([=]() {});
        });
      }

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

} /* namespace TEST_NAMESPACE */

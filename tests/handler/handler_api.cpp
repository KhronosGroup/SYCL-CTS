/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME handler_api

namespace handler_api__ {
using namespace sycl_cts;

/** tests the api for cl::sycl::handler
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  virtual void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  virtual void run(util::logger &log) override {
    try {
      cl::sycl::queue queue;

      int results[5];

      {
        cl::sycl::range<1> range(1);
        cl::sycl::buffer<int, 1> buffer(results, cl::sycl::range<1>(5));

        /** check set_arg() methods
        */
        queue.submit([&](cl::sycl::handler &cgh) {
          auto accessor =
              buffer.get_access<cl::sycl::access::mode::read_write,
                                cl::sycl::access::target::global_buffer>(cgh);
          cgh.set_arg(0, accessor);

          int scalar = 5;
          cgh.set_arg(1, scalar);

          cl::sycl::sampler sampler(true,
                                    cl::sycl::info::addressing_mode::clamp,
                                    cl::sycl::info::filter_mode::nearest);
          cgh.set_arg(2, sampler);
        });

        /** check single_task() method
        */
        queue.submit([&](cl::sycl::handler &cgh) {
          auto accessor =
              buffer.get_access<cl::sycl::access::mode::write,
                                cl::sycl::access::target::global_buffer>(cgh);

          cgh.single_task<class a_single_task>([=] { accessor[0] = 1; });
        });

        /** check parallel_for() method
        */
        queue.submit([&](cl::sycl::handler &cgh) {
          auto accessor =
              buffer.get_access<cl::sycl::access::mode::write,
                                cl::sycl::access::target::global_buffer>(cgh);

          cgh.parallel_for<class a_parallel_for>(
              range<1>(1), [=](id<1> in_id) { accessor[1] = 1; });

        });

        /** check parallel_for() method
        */
        queue.submit([&](cl::sycl::handler &cgh) {
          auto accessor =
              buffer.get_access<cl::sycl::access::mode::write,
                                cl::sycl::access::target::global_buffer>(cgh);

          cgh.parallel_for<class a_parallel_for>(
              range<1>(1), [=](item<1> in_id) { accessor[2] = 1; });

        });

        /** check parallel_for() method
        */
        queue.submit([&](cl::sycl::handler &cgh) {
          auto accessor =
              buffer.get_access<cl::sycl::access::mode::write,
                                cl::sycl::access::target::global_buffer>(cgh);

          cgh.parallel_for<class a_parallel_for>(
              range<1>(1), [=](item<1> in_id) { accessor[3] = 1; });

        });

        /** check parallel_for_work_group() method
        */
        queue.submit([&](cl::sycl::handler &cgh) {
          auto accessor =
              buffer.get_access<cl::sycl::access::mode::write,
                                cl::sycl::access::target::global_buffer>(cgh);

          cl::sycl::nd_range<1> wg_range =
              nd_range<1>(range<1>(1), range<1>(1));
          cgh.parallel_for_work_group<class a_parallel_for_worgroup>(
              wg_range, [=](cl::sycl::group<1> in_group) { accessor[4] = 1; });

        });
      }

      if ((results[0] & results[1] & results[2] & results[3] & results[4]) ==
          0) {
        FAIL(log, "handler did not execute all kernels");
      }

      queue.wait_and_throw();
    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      FAIL(log, "a sycl exception was caught");
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace handler_api__ */

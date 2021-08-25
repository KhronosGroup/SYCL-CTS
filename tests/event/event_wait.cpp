/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME event_wait

namespace TEST_NAMESPACE {

using namespace sycl_cts;

class add_kernel;
class mul_kernel;

/** test the wait api for sycl::event
 */
class TEST_NAME : public sycl_cts::util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /* enqueue an add command and return the complete event */
  sycl::event add_operation(sycl_cts::util::logger &log, sycl::queue &queue,
                            sycl::buffer<float, 1> &d_data,
                            const float operand) {
    return queue.submit([&](sycl::handler &cgh) {
      auto a_data = d_data.get_access<sycl::access_mode::read_write>(cgh);

      cgh.single_task<class add_kernel>([=]() { a_data[0] += operand; });
    });
  }

  /* enqueue a mul command and return the complete event */
  sycl::event mul_operation(sycl_cts::util::logger &log, sycl::queue &queue,
                            sycl::buffer<float, 1> &d_data,
                            const float operand) {
    return queue.submit([&](sycl::handler &cgh) {
      auto a_data = d_data.get_access<sycl::access_mode::read_write>(cgh);

      cgh.single_task<class mul_kernel>([=]() { a_data[0] *= operand; });
    });
  }

  /** Execute kernels, waiting in-between
   */
  bool wait_and_exec(sycl_cts::util::logger &log, sycl::queue &queueA,
                     sycl::queue &queueB) {
    for (int i = 0; i < 4; ++i) {
      float h_data = 1.0;
      {  // Create a new scope so we can check the result of the buffer when
        // it's written back to host

        sycl::buffer<float, 1> d_data(&h_data, sycl::range<1>(1));

        sycl::event complete = mul_operation(log, queueA, d_data, 2.0);

        switch (i) {
          case 0: {  // Test sycl::event::wait()
            complete.wait();
            break;
          }
          case 1: {  // Test sycl::event::wait_and_throw()
            complete.wait_and_throw();
            break;
          }
          case 2: {  // Test sycl::event::wait(std::vector<event>)
            std::vector<sycl::event> evt_list = complete.get_wait_list();
            sycl::event::wait(evt_list);
            break;
          }
          case 3: {  // Test
            // sycl::event::wait_and_throw(std::vector<event>)
            std::vector<sycl::event> evt_list = complete.get_wait_list();
            sycl::event::wait_and_throw(evt_list);
            break;
          }
        }
        add_operation(log, queueB, d_data, -3.0);
      }
      if (h_data != -1.0) {
        return false;
      }
    }

    return true;
  }

  /** execute the test
   */
  void run(sycl_cts::util::logger &log) override {
    auto queueA = util::get_cts_object::queue();
    auto queueB = util::get_cts_object::queue();

    if (!wait_and_exec(log, queueA, queueB)) {
      FAIL(log, "sycl::event::wait() tests failed");
    }

    queueA.wait_and_throw();
    queueB.wait_and_throw();
  }
};

// construction of this proxy will register the above test
sycl_cts::util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */

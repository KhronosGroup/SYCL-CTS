/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME event_wait

namespace TEST_NAMESPACE {

using namespace sycl_cts;

/** test the wait api for cl::sycl::event
*/
class TEST_NAME : public sycl_cts::util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /* enqueue an add command and return the complete event */
  cl::sycl::event add_operation(sycl_cts::util::logger &log,
    cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &d_data,
    const float operand) {
    return queue.submit([&](cl::sycl::handler &cgh) {
      auto a_data = d_data.get_access<cl::sycl::access::mode::read_write>(cgh);

      cgh.single_task<class add_kernel>([=]() { a_data[0] += operand; });
    });
  }

  /* enqueue a mul command and return the complete event */
  cl::sycl::event mul_operation(sycl_cts::util::logger &log,
    cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &d_data,
    const float operand) {
    return queue.submit([&](cl::sycl::handler &cgh) {
      auto a_data = d_data.get_access<cl::sycl::access::mode::read_write>(cgh);

      cgh.single_task<class mul_kernel>([=]() { a_data[0] *= operand; });

    });
  }

  /** Execute kernels, waiting in-between
   */
  bool wait_and_exec(sycl_cts::util::logger &log, cl::sycl::queue &queueA,
                     cl::sycl::queue &queueB) {
    for (int i = 0; i < 4; ++i) {
      float h_data = 1.0;
      {  // Create a new scope so we can check the result of the buffer when
        // it's written back to host

        cl::sycl::buffer<float, 1> d_data(&h_data, cl::sycl::range<1>(1));

        cl::sycl::event complete = mul_operation(log, queueA, d_data, 2.0);

        switch (i) {
          case 0: {  // Test cl::sycl::event::wait()
            complete.wait();
            break;
          }
          case 1: {  // Test cl::sycl::event::wait_and_throw()
            complete.wait_and_throw();
            break;
          }
          case 2: {  // Test cl::sycl::event::wait(vector_class<event>)
            cl::sycl::vector_class<cl::sycl::event> evt_list =
              complete.get_wait_list();
            cl::sycl::event::wait(evt_list);
            break;
          }
          case 3: {  // Test
            // cl::sycl::event::wait_and_throw(vector_class<event>)
            cl::sycl::vector_class<cl::sycl::event> evt_list =
              complete.get_wait_list();
            cl::sycl::event::wait_and_throw(evt_list);
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
    try {
      auto queueA = util::get_cts_object::queue();
      auto queueB = util::get_cts_object::queue();

      if (!wait_and_exec(log, queueA, queueB)) {
        FAIL(log, "cl::sycl::event::wait() tests failed");
      }

      queueA.wait_and_throw();
      queueB.wait_and_throw();
    } catch (const cl::sycl::exception &e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// construction of this proxy will register the above test
sycl_cts::util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */

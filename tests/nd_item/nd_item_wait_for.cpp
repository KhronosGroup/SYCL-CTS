/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME nd_item_wait_for

namespace nd_item_wait_for__ {
using namespace sycl_cts;

class wait_for_kernel;
void test_wait_for(util::logger &log, cl::sycl::queue &queue) {
  /* set workspace size */
  const int globalSize = 64;
  const int localSize = 2;

  /* allocate and assign host data */

  /* init ranges*/
  cl::sycl::range<1> globalRange(globalSize);
  cl::sycl::range<1> localRange(localSize);
  cl::sycl::nd_range<1> NDRange(globalRange, localRange);

  /* run kernel to check wait_for interface is available*/
  int data = 1234;
  {
    auto buf = cl::sycl::buffer<int, 1>(&data, cl::sycl::range<1>(1));

    queue.submit([&](cl::sycl::handler &cgh) {
      auto globalAcc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
      auto localAcc =
          cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                             cl::sycl::access::target::local>(
              cl::sycl::range<1>(1), cgh);

      cgh.parallel_for<class wait_for_kernel>(
          NDRange, [=](cl::sycl::nd_item<1> item) {

            cl::sycl::device_event device_event = item.async_work_group_copy(
                localAcc.get_pointer(), globalAcc.get_pointer(), 1);

            item.wait_for(device_event);

          });
    });
  }
}

/** test cl::sycl::nd_item wait_for
*/
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
  *  @param info, test_base::info structure as output
  */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
  *  @param log, test transcript logging class
  */
  void run(util::logger &log) override {
    try {
      auto cmdQueue = util::get_cts_object::queue();

      test_wait_for(log, cmdQueue);

      cmdQueue.wait_and_throw();
    } catch (const cl::sycl::exception &e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

util::test_proxy<TEST_NAME> proxy;

} /* namespace nd_item_wait_for__ */

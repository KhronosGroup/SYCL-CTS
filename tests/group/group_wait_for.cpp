/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME group_wait_for

namespace group_wait_for__ {
using namespace sycl_cts;

void test_wait_for(util::logger &log, cl::sycl::queue &queue) {
  /* set workspace size */
  const int globalSize = 64;
  const int localSize = 2;

  /* allocate and assign host data */

  /* init ranges*/
  cl::sycl::range<1> globalRange(globalSize);

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

      cgh.parallel_for_work_group<class TEST_NAME>(
          globalRange, [=](cl::sycl::group<1> test_group) {

            cl::sycl::device_event device_event_1 =
                test_group.async_work_group_copy(localAcc.get_pointer(),
                                                 globalAcc.get_pointer(), 1);

            test_group.wait_for(device_event_1);

            cl::sycl::device_event device_event_2 =
                test_group.async_work_group_copy(localAcc.get_pointer(),
                                                 globalAcc.get_pointer(), 1);

            cl::sycl::device_event device_event_3 =
                test_group.async_work_group_copy(localAcc.get_pointer(),
                                                 globalAcc.get_pointer(), 1);

            test_group.wait_for(device_event_2, device_event_3);

          });
    });
  }
}

/** test cl::sycl::group::wait_for
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

} /* namespace group_wait_for__ */

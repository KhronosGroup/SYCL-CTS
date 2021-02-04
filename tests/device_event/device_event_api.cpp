/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME device_event_api

namespace device_event_api__ {
using namespace sycl_cts;

class device_event_wait;

/** tests the api for cl::sycl::device_event
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
      /** check wait() member function
       */
      {
        auto testQueue = util::get_cts_object::queue();

        int data = 1234;
        {
          auto buf = cl::sycl::buffer<int, 1>(&data, cl::sycl::range<1>(1));

          testQueue.submit([&](cl::sycl::handler &cgh) {
            auto globalAcc =
                buf.get_access<cl::sycl::access::mode::read_write>(cgh);
            auto localAcc =
                cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                                   cl::sycl::access::target::local>(
                    cl::sycl::range<1>(1), cgh);

            cgh.parallel_for<class device_event_wait>(
                cl::sycl::nd_range<1>(cl::sycl::range<1>(1),
                                      cl::sycl::range<1>(1)),
                [=](cl::sycl::nd_item<1> ndItem) {
                  cl::sycl::device_event deviceEvent =
                      ndItem.async_work_group_copy(localAcc.get_pointer(),
                                                   globalAcc.get_pointer(), 1);

                  deviceEvent.wait();
                });
          });
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

} /* namespace context_api */

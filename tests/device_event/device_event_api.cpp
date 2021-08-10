/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#include <algorithm>
#include <array>
#include <string>

#define TEST_NAME device_event_api

namespace TEST_NAMESPACE {
using namespace sycl_cts;

class device_event_wait;

/** tests the api for sycl::device_event
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
      auto testQueue = util::get_cts_object::queue();

      constexpr size_t bufferSize = 512;
      constexpr size_t sampleIndex = bufferSize / 2;
      constexpr int referenceValue = 1234;

      std::array<int, bufferSize> data;
      std::fill(data.begin(), data.end(), referenceValue);

      bool error = false;
      {
        sycl::range<1> range(1);
        sycl::range<1> dataRange(bufferSize);
        sycl::buffer<int, 1> buf(data.data(), dataRange);
        sycl::buffer<bool, 1> errBuf(&error, range);

        testQueue.submit([&](sycl::handler &cgh) {

          auto globalAcc = buf.get_access<sycl::access_mode::read_write>(cgh);
          auto errorAcc = errBuf.get_access<sycl::access_mode::write>(cgh);
          auto localAcc =
              sycl::accessor<int, 1, sycl::access_mode::read_write,
                  sycl::target::local>(dataRange, cgh);

          cgh.parallel_for<class device_event_wait>(
              sycl::nd_range<1>(range, range),
              [=](sycl::nd_item<1> ndItem) {
                // Run asynchronous copy for full buffer
                sycl::device_event deviceEvent =
                    ndItem.async_work_group_copy(localAcc.get_pointer(),
                                                 globalAcc.get_pointer(),
                                                 bufferSize);

                deviceEvent.wait();

                // Check sample was updated
                if (localAcc[sampleIndex] != referenceValue) {
                  errorAcc[0] = true;
                }
              });
        });
      }
      if (error) {
        FAIL(log, "sycl::device_event async_work_group_copy failed");
      }
    } catch (const sycl::exception &e) {
      log_exception(log, e);
      const auto errorMsg =
          std::string("a SYCL exception was caught: ") + e.what();
      FAIL(log, errorMsg);
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE

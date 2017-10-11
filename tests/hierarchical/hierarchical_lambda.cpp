/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME hierarchical_lambda

namespace TEST_NAMESPACE {

using namespace sycl_cts;

/** test cl::sycl::range::get(int index) return size_t
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
      constexpr unsigned int globalRange1d = 6;
      constexpr unsigned int globalRange2d = 2;
      constexpr unsigned int local = globalRange2d;
      std::vector<int> data(globalRange1d * globalRange2d, 0);

      auto myQueue = util::get_cts_object::queue();
      // using this scope we ensure that the buffer will update the host values
      // after the wait_and_throw
      {
        cl::sycl::buffer<int, 1> buf(
            data.data(), cl::sycl::range<1>(globalRange1d * globalRange2d));

        myQueue.submit([&](cl::sycl::handler &cgh) {
          auto globalRange = cl::sycl::range<2>(globalRange1d, globalRange2d);
          auto localRange = cl::sycl::range<2>(local, local);
          auto groupRange = globalRange / localRange;
          auto ptr =
              buf.get_access<cl::sycl::access::mode::read_write,
                             cl::sycl::access::target::global_buffer>(cgh);
          cgh.parallel_for_work_group<class hierarchical_lambda>(
              groupRange, localRange, [ptr](cl::sycl::group<2> group_pid) {

                parallel_for_work_item(
                    group_pid, [group_pid, ptr](cl::sycl::item<2> itemID) {
                      auto localId = itemID.get_id();
                      auto localSize = itemID.get_range();
                      auto globalId = group_pid.get() * localSize + localId;
                      int globalIdL =
                          ((globalId.get(0) * 2 * 1) + globalId.get(1));
                      ptr[globalIdL] = globalIdL;
                    });
              });
        });
        myQueue.wait_and_throw();
      }

      for (size_t i = 0; i < globalRange1d * globalRange2d; i++) {
        if (data[i] != i) {
          cl::sycl::string_class errorMessage =
              cl::sycl::string_class("Value for global id ") +
              std::to_string(i) + cl::sycl::string_class(" was not correct (") +
              std::to_string(data[i]) + cl::sycl::string_class(" instead of ") +
              std::to_string(i);
          FAIL(log, errorMessage);
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

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE

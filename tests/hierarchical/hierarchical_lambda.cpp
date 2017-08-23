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
using namespace cl::sycl;

/** test cl::sycl::range::get(int index) return size_t
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
      int data = 1;
      int expected = 4096;

      auto myQueue = util::get_cts_object::queue();
      buffer<int, 1> buf(&data, range<1>(1));
      myQueue.submit([&](handler &cgh) {
        auto GlobalRange = range<2>(6, 2);
        auto LocalRange = range<2>(2, 2);
        auto GroupRange = GlobalRange / LocalRange;

        accessor<int, 1, cl::sycl::access::mode::read_write,
                 cl::sycl::access::target::global_buffer>
            ptr = buf.get_access<cl::sycl::access::mode::read_write,
                                 cl::sycl::access::target::global_buffer>(cgh);

        cgh.parallel_for_work_group<class kernel0>(
            GroupRange, LocalRange, ([=](group<2> group_id) {
              parallel_for_work_item(group_id,
                                     [ptr](item<2> item_id) { ptr[0] *= 2; });
            }));
      });
      myQueue.wait_and_throw();

      accessor<int, 1, cl::sycl::access::mode::read,
               cl::sycl::access::target::host_buffer>
          hostPtr(buf);
      if (hostPtr[0] != expected) {
        FAIL(log, "Value not as expected.");
      }

    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace hierarchical_lambda__ */

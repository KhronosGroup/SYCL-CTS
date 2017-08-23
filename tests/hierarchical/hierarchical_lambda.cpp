/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME hierarchical_lambda

namespace hierarchical_lambda__ {

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
      default_selector sel;
      int data = 1;
      int expected = 4096;

      queue my_queue(sel);
      buffer<int, 1> buf(&data, range<1>(1));
      my_queue.submit([&](handler &cgh) {
        auto my_range = nd_range<2>(range<2>(6, 2), range<2>(2, 2));

        accessor<int, 1, cl::sycl::access::mode::read_write,
                 cl::sycl::access::target::global_buffer>
            ptr(buf, cgh);

        cgh.parallel_for_work_group<class kernel0>(
            my_range, ([=](group<2> group_id) {
              parallel_for_work_item(group_id,
                                     [=](item<2> item_id) { ptr[0] *= 2; });
            }));
      });

      accessor<int, 1, cl::sycl::access::mode::read,
               cl::sycl::access::target::host_buffer>
          host_ptr(buf);
      if (host_ptr[0] != expected) {
        FAIL(log, "Value not as expected.");
      }

      my_queue.wait_and_throw();

    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      FAIL(log, "sycl exception caught");
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace id_api__ */

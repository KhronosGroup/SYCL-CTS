/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME hierarchical_functor

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace cl::sycl;

class kernel0 {
  accessor<int, 1, cl::sycl::access::mode::read_write,
           cl::sycl::access::target::global_buffer>
      ptr;

 public:
  kernel0(accessor<int, 1, cl::sycl::access::mode::read_write,
                   cl::sycl::access::target::global_buffer>
              p)
      : ptr(p) {}

  void operator()(group<2> group_pid) const {
    parallel_for_work_item(group_pid, [=](item<2> itemID) { ptr[0] *= 2; });
  }
};

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
        auto globalRange = range<2>(6, 2);
        auto localRange = range<2>(2, 2);
        auto groupRange = globalRange / localRange;

        accessor<int, 1, cl::sycl::access::mode::read_write,
                 cl::sycl::access::target::global_buffer>
            ptr(buf, cgh);
        cgh.parallel_for_work_group(groupRange, localRange, kernel0(ptr));
      });
      myQueue.wait_and_throw();
      accessor<int, 1, cl::sycl::access::mode::read,
               cl::sycl::access::target::host_buffer>
          host_ptr(buf);
      if (host_ptr[0] != expected) {
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

} /* namespace id_api__ */

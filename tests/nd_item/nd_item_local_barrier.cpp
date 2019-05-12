/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME nd_item_local_barrier

namespace nd_item_local_barrier__ {
using namespace sycl_cts;

class local_barrier_kernel;
void test_barrier(util::logger &log, cl::sycl::queue &queue) {
  /* set workspace size */
  const int globalSize = 64;
  const int localSize = 2;

  /* allocate and assign host data */
  std::unique_ptr<int> data(new int[globalSize]);

  for (int i = 0; i < globalSize; ++i) data.get()[i] = i;

  /* init ranges*/
  cl::sycl::range<1> globalRange(globalSize);
  cl::sycl::range<1> localRange(localSize);
  cl::sycl::nd_range<1> NDRange(globalRange, localRange);

  /* run kernel to swap adjacent work item's global id*/
  {
    cl::sycl::buffer<int, 1> buf(data.get(), globalRange);

    queue.submit([&](cl::sycl::handler &cgh) {
      auto accGlobal = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
      cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::local>
          localScratch(localRange, cgh);

      cgh.parallel_for<class local_barrier_kernel>(
          NDRange, [=](cl::sycl::nd_item<1> item) {
            int idx = (int)item.get_global_id(0);
            int pos = idx & 1;
            int opp = pos ^ 1;

            localScratch[pos] = accGlobal[idx];

            item.barrier(cl::sycl::access::fence_space::local_space);

            accGlobal[idx] = localScratch[opp];
          });
    });
  }

  /* check correct results returned*/
  bool passed = true;
  for (int i = 0; i < globalSize; ++i) {
    if (i % 2 == 0)
      passed &= data.get()[i] == (i + 1);
    else
      passed &= data.get()[i] == (i - 1);
  }

  if (!passed) {
    FAIL(log, "local barrier failed");
  }
}

/** test cl::sycl::nd_item local barrier
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

      test_barrier(log, cmdQueue);

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

} /* namespace nd_item_local_barrier__ */

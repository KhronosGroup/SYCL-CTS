/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME nd_item_local_barrier

namespace nd_item_local_barrier__ {
using namespace cl::sycl;
using namespace sycl_cts;

void test_barrier(util::logger &log, cl::sycl::queue &queue) {
  /* set workspace size */
  const int globalSize = 64;
  const int localSize = 2;

  /* allocate and assign host data */
  std::unique_ptr<int> data(new int[globalSize]);

  for (int i = 0; i < globalSize; ++i) data.get()[i] = i;

  /* init ranges*/
  range<1> globalRange(globalSize);
  range<1> localRange(localSize);
  nd_range<1> NDRange(globalRange, localRange);

  /* run kernel to swap adjancent work item's global id*/
  {
    buffer<int, 1> buf(data.get(), globalRange);

    queue.submit([&](handler &cgh) {
      auto accGlobal = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
      accessor<int, 1, cl::sycl::access::mode::read_write,
               cl::sycl::access::target::local> localScratch(localRange, cgh);

      cgh.parallel_for<class local_barrier_kernel>(NDRange,
                                                   [=](nd_item<1> item) {
        int idx = (int)item.get_global(0);
        int pos = idx & 1;
        int opp = pos ^ 1;

        localScratch[pos] = accGlobal[idx];

        item.barrier(access::fence_space::local);

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
  virtual void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
  *  @param log, test transcript logging class
  */
  virtual void run(util::logger &log) override {
    try {
      cts_selector selector;
      queue cmdQueue(selector);

      test_barrier(log, cmdQueue);

      cmdQueue.wait_and_throw();

    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      FAIL(log, "sycl exception caught");
    }
  }
};

util::test_proxy<TEST_NAME> proxy;

} /* namespace nd_item_local_barrier__ */

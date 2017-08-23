/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME nd_item_combined_barrier

namespace nd_item_combined_barrier__ {
using namespace cl::sycl;
using namespace sycl_cts;

void test_barrier(util::logger &log, cl::sycl::queue &queue) {
  /* set workspace size */
  const int globalSize = 64;
  const int localSize = 2;

  /* allocate and assign host data */
  std::unique_ptr<int> globalID(new int[globalSize]);
  std::unique_ptr<int> localID(new int[globalSize]);
  std::unique_ptr<int> globScratch(new int[globalSize]);

  for (int i = 0; i < globalSize; ++i) {
    globalID.get()[i] = i;
    localID.get()[i] = i % localSize;
    globScratch.get()[i] = 0;
  }

  /* init ranges*/
  range<1> globalRange(globalSize);
  range<1> localRange(localSize);
  nd_range<1> NDRange(globalRange, localRange);

  /* run kernel to swap adjancent work item's global & local ids*/
  {
    buffer<int, 1> globalBuf(globalID.get(), globalRange);
    buffer<int, 1> scratchBuf(globScratch.get(), globalRange);
    buffer<int, 1> localBuf(localID.get(), globalRange);

    queue.submit([&](handler &cgh) {
      auto accGlobal =
          globalBuf.get_access<cl::sycl::access::mode::read_write>(cgh);
      auto accScratchGlobal =
          scratchBuf.get_access<cl::sycl::access::mode::read_write>(cgh);
      auto accLocal =
          localBuf.get_access<cl::sycl::access::mode::read_write>(cgh);
      accessor<int, 1, cl::sycl::access::mode::read_write,
               cl::sycl::access::target::local>
          accScratchLocal(localRange, cgh);

      cgh.parallel_for<class combined_barrier_kernel>(
          NDRange, [=](nd_item<1> item) {
            int idx = (int)item.get_global(0);
            int pos = idx & 1;
            int opp = pos ^ 1;

            accScratchGlobal[pos] = accGlobal[idx];
            accScratchLocal[pos] = accLocal[idx];

            item.barrier(access::fence_space::global_and_local);

            accLocal[idx] = accScratchLocal[opp];
            accGlobal[idx] = accScratchGlobal[opp];

          });
    });
  }

  /* check correct results returned*/
  bool passed = true;
  for (int i = 0; i < globalSize; ++i) {
    if (i % 2 == 0)
      passed &= globalID.get()[i] == (i + 1) && localID.get()[i] == 1;
    else
      passed &= globalID.get()[i] == (i - 1) && localID.get()[i] == 0;
  }

  if (!passed) {
    FAIL(log, "global_and_local barrier failed");
  }
}

/** test cl::sycl::nd_item barrier functions
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

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace nd_item_combined_barrier__

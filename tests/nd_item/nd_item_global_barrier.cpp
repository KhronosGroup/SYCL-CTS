/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME nd_item_global_barrier

namespace nd_item_global_barrier__ {
using namespace cl::sycl;
using namespace sycl_cts;

void test_barrier(util::logger &log, cl::sycl::queue &queue) {
  /* set workspace size */
  const int globalSize = 64;

  /* allocate and assign host data */
  std::unique_ptr<int[]> data(new int[globalSize]);

  for (int i = 0; i < globalSize; ++i) {
    data.get()[i] = i;
  }

  /* run kernel to swap adjacent work item's global & local ids*/
  {
    buffer<int, 1> buffer(data.get(), range<1>(globalSize));

    queue.submit([&](handler &cgh) {
      auto ptr = buffer.get_access<access::mode::read_write>(cgh);

      accessor<int, 1, access::mode::read_write, access::target::local> tile(
          range<1>(2), cgh);

      cgh.parallel_for<class global_barrier_kernel>(
          nd_range<1>(range<1>(64), range<1>(2)), [=](nd_item<1> item) {
            size_t idx = item.get_global_linear_id();
            size_t pos = idx & 1;
            size_t opp = pos ^ 1;

            tile[pos] = ptr[idx];

            item.barrier(access::fence_space::global_space);

            ptr[idx] = tile[opp];
          });
    });
  }

  /* check correct results returned*/
  bool passed = true;
  for (int i = 0; i < globalSize; i += 2) {
    int current = i;
    int next = i + 1;
    if ((data[current] != next) || (data[next] != current)) {
      passed = false;
    }
  }

  if (!passed) {
    FAIL(log, "global barrier failed");
  }
}

/** test cl::sycl::nd_item global barrier
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

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace nd_item_global_barrier__ */

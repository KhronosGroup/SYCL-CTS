/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"
#include "../nd_item/nd_item_barrier_common.h"

#define TEST_NAME nd_item_global_barrier

namespace nd_item_global_barrier__ {
using namespace sycl_cts;

template <int dim>
class global_barrier_kernel_fence;

template<int dim>
struct barrierCall {
    void operator()(sycl::nd_item<dim> item) const {
      item.barrier(sycl::access::fence_space::global_space);
    }
};

/** test sycl::nd_item global barrier
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
    {
      auto cmdQueue = util::get_cts_object::queue();

      // Verify global barrier works as fence for global address space
      std::string errorMsg =
          "global barrier failed for global address space";
      test_barrier_global_space<1, global_barrier_kernel_fence<1>>(
          log, cmdQueue, barrierCall<1>(), errorMsg);
      test_barrier_global_space<2, global_barrier_kernel_fence<2>>(
          log, cmdQueue, barrierCall<2>(), errorMsg);
      test_barrier_global_space<3, global_barrier_kernel_fence<3>>(
          log, cmdQueue, barrierCall<3>(), errorMsg);

      cmdQueue.wait_and_throw();
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace nd_item_global_barrier__ */

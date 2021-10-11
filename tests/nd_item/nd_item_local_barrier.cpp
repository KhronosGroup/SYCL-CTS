/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"
#include "../nd_item/nd_item_barrier_common.h"

#define TEST_NAME nd_item_local_barrier

namespace nd_item_local_barrier__ {
using namespace sycl_cts;

template <int dim>
class local_barrier_kernel_fence;

template<int dim>
struct barrierCall {
    void operator()(sycl::nd_item<dim> item) const {
      item.barrier(sycl::access::fence_space::local_space);
    }
};

template <class kernelT, typename barrierCallT, int dim>
void test_barrier_local_space_all_dims(std::string errorMsg) {

}

/** test sycl::nd_item local barrier
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

      // Verify local barrier works as fence for local address space
      std::string errorMsg =
          "local barrier failed for local address space";
      test_barrier_local_space<1, local_barrier_kernel_fence<1>>(
          log, cmdQueue, barrierCall<1>(), errorMsg);
      test_barrier_local_space<2, local_barrier_kernel_fence<2>>(
          log, cmdQueue, barrierCall<2>(), errorMsg);
      test_barrier_local_space<3, local_barrier_kernel_fence<3>>(
          log, cmdQueue, barrierCall<3>(), errorMsg);

      cmdQueue.wait_and_throw();
    }
  }
};

util::test_proxy<TEST_NAME> proxy;

} /* namespace nd_item_local_barrier__ */

/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"
#include "../nd_item/nd_item_barrier_common.h"

#define TEST_NAME nd_item_default_barrier

namespace nd_item_default_barrier__ {
using namespace sycl_cts;

template <int dim>
class default_barrier_kernel_local;
template <int dim>
class default_barrier_kernel_global;

template<int dim>
struct barrierCall {
    void operator()(sycl::nd_item<dim> item) const {
      item.barrier();
    }
};


/** test sycl::nd_item barrier functions
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

      // Verify default barrier works as fence for local address space
      std::string errorMsg =
          "default barrier failed for local address space";
      test_barrier_local_space<1, default_barrier_kernel_local<1>>(
          log, cmdQueue, barrierCall<1>(), errorMsg);
      test_barrier_local_space<2, default_barrier_kernel_local<2>>(
          log, cmdQueue, barrierCall<2>(), errorMsg);
      test_barrier_local_space<3, default_barrier_kernel_local<3>>(
          log, cmdQueue, barrierCall<3>(), errorMsg);

      // Verify default barrier works as fence for global address space
      errorMsg =
          "default barrier failed for global address space";
      test_barrier_global_space<1, default_barrier_kernel_global<1>>(
          log, cmdQueue, barrierCall<1>(), errorMsg);
      test_barrier_global_space<2, default_barrier_kernel_global<2>>(
          log, cmdQueue, barrierCall<2>(), errorMsg);
      test_barrier_global_space<3, default_barrier_kernel_global<3>>(
          log, cmdQueue, barrierCall<3>(), errorMsg);

      cmdQueue.wait_and_throw();
    } catch (const sycl::exception &e) {
      log_exception(log, e);
      std::string errorMsg =
          "a SYCL exception was caught: " + std::string(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace nd_item_default_barrier__

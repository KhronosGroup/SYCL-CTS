/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/async_work_group_copy.h"
#include "../common/invoke.h"

#define TEST_NAME nd_item_wait_for

namespace TEST_NAMESPACE {
using namespace sycl_cts;

template <int dim>
class wait_for_kernel;

template<int dim>
struct check_dim {
  void operator()(sycl::queue &queue, sycl_cts::util::logger &log) {
    using dataT = int;
    using kernelT = wait_for_kernel<dim>;
    using kernelInvokeT = invoke_nd_item<dim, kernelT>;
    static const std::string instanceName = "nd_item";

    test_wait_for<kernelInvokeT, dataT>(queue, log, instanceName);
  }
};

/** test sycl::nd_item wait_for
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
      auto queue = util::get_cts_object::queue();

      check_all_dims<check_dim>(queue, log);
    }
  }
};

util::test_proxy<TEST_NAME> proxy;

} /* namespace nd_item_wait_for__ */

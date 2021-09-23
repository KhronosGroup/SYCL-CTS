/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME hierarchical_lambda

namespace TEST_NAMESPACE {

template <int dim>
class kernel;

using namespace sycl_cts;

template <int dim>
void check_dim(util::logger &log) {
  constexpr size_t globalRange1d = 6;
  constexpr size_t globalRange2d = 2;
  constexpr size_t globalRangeTotal = 24;
  constexpr size_t local = globalRange2d;
  std::vector<size_t> data(globalRangeTotal, 0);

  auto myQueue = util::get_cts_object::queue();
  // using this scope we ensure that the buffer will update the host values
  // after the wait_and_throw
  {
    sycl::buffer<size_t, 1> buf(data.data(), sycl::range<1>(globalRangeTotal));

    myQueue.submit([&](sycl::handler &cgh) {
      auto globalRange =
          sycl_cts::util::get_cts_object::range<dim>::template get_fixed_size<
              globalRangeTotal>(globalRange1d, globalRange2d);
      auto localRange =
          sycl_cts::util::get_cts_object::range<dim>::get(local, local, local);
      auto groupRange = globalRange / localRange;
      auto ptr =
          buf.get_access<sycl::access_mode::read_write, sycl::target::device>(
              cgh);
      cgh.parallel_for_work_group<kernel<dim>>(
          groupRange, localRange, [ptr](sycl::group<dim> group_pid) {
            // Assign global linear id to captured varible
            group_pid.parallel_for_work_item([ptr](sycl::h_item<dim> itemID) {
              auto globalIdL = itemID.get_global().get_linear_id();
              ptr[globalIdL] = globalIdL;
            });
          });
    });
  }

  for (size_t i = 0; i < globalRangeTotal; i++) {
    if (data[i] != i) {
      std::string errorMessage =
          std::string("Value for global id ") + std::to_string(i) +
          std::string(" was not correct (") + std::to_string(data[i]) +
          std::string(" instead of ") + std::to_string(i);
      FAIL(log, errorMessage);
    }
  }
}

/** test sycl::range::get(int index) return size_t
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  void run(util::logger &log) override {
    check_dim<1>(log);
    check_dim<2>(log);
    check_dim<3>(log);
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE

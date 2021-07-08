/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME hierarchical_functor

namespace TEST_NAMESPACE {
using namespace sycl_cts;

template <int dim> class kernel {
  sycl::accessor<size_t, 1, sycl::access::mode::read_write,
                     sycl::target::global_buffer> ptr;

 public:
   kernel(sycl::buffer<size_t, 1> buf, sycl::handler &cgh)
       : ptr(buf.get_access<sycl::access::mode::read_write,
                            sycl::target::global_buffer>(cgh)) {}

   void operator()(sycl::group<dim> group_pid) const {
     group_pid.parallel_for_work_item([&](sycl::h_item<dim> itemID) {
       auto globalIdL = itemID.get_global().get_linear_id();
       ptr[globalIdL] = globalIdL;
     });
  }
};

template <int dim> void check_dim(util::logger &log) {
  constexpr size_t globalRange1d = 8;
  constexpr size_t globalRange2d = 2;
  constexpr size_t totalGlobalRange = 64;
  constexpr size_t local = globalRange2d;
  std::vector<size_t> data(totalGlobalRange, 0);

  auto myQueue = util::get_cts_object::queue();
  // using this scope we ensure that the buffer will update the host values
  // after the wait_and_throw
  {
    sycl::buffer<size_t, 1> buf(data.data(),
                                    sycl::range<1>(totalGlobalRange));

    myQueue.submit([&](sycl::handler &cgh) {
      auto globalRange =
          sycl_cts::util::get_cts_object::range<dim>::template get_fixed_size<
              totalGlobalRange>(globalRange1d, globalRange2d);
      auto localRange =
          sycl_cts::util::get_cts_object::range<dim>::get(local, local, local);
      auto groupRange = globalRange / localRange;

      // Assign global linear item's id in kernel functor
      cgh.parallel_for_work_group(groupRange, localRange,
                                  kernel<dim>(buf, cgh));
    });
  }
  for (size_t i = 0; i < totalGlobalRange; i++) {
    if (data[i] != i) {
      sycl::string_class errorMessage =
          sycl::string_class("Value for global id ") + std::to_string(i) +
          sycl::string_class(" was not correct (") +
          std::to_string(data[i]) + sycl::string_class(" instead of ") +
          std::to_string(i) + sycl::string_class(". dim = ") +
          std::to_string(dim);
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
    try {
      check_dim<1>(log);
      check_dim<2>(log);
      check_dim<3>(log);
    } catch (const sycl::exception &e) {
      log_exception(log, e);
      sycl::string_class errorMsg =
          "a SYCL exception was caught: " + sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace id_api__ */

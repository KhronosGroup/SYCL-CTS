/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME nd_range_api

namespace TEST_NAMESPACE {
using namespace sycl_cts;

static const size_t sizes[] = {16, 32, 64};

template <int dim>
void test_nd_range(util::logger &log, sycl::range<dim> gs,
                   sycl::range<dim> ls) {
  for (int i = 0; i < dim; i++) {
    sycl::nd_range<dim> nd_range(gs, ls);
    CHECK_TYPE(log, nd_range.get_global_range()[i], sizes[i]);
    CHECK_VALUE(log, nd_range.get_global_range()[i], sizes[i], i);
    CHECK_TYPE(log, nd_range.get_local_range()[i], sizes[i] / 4);
    CHECK_VALUE(log, nd_range.get_local_range()[i], sizes[i] / 4, i);

    CHECK_TYPE(log, nd_range.get_group_range()[i], sizes[i] / (sizes[i] / 4));
    CHECK_VALUE(log, nd_range.get_group_range()[i], sizes[i] / (sizes[i] / 4), i);

    sycl::nd_range<dim> deep_copy(nd_range);

    CHECK_TYPE(log, deep_copy.get_global_range()[i], sizes[i]);
    CHECK_VALUE(log, deep_copy.get_global_range()[i], sizes[i], i);
    CHECK_TYPE(log, deep_copy.get_local_range()[i], sizes[i] / 4);
    CHECK_VALUE(log, deep_copy.get_local_range()[i], sizes[i] / 4, i);

    CHECK_TYPE(log, deep_copy.get_group_range()[i], sizes[i] / (sizes[i] / 4));
    CHECK_VALUE(log, deep_copy.get_group_range()[i], sizes[i] / (sizes[i] / 4), i);

  }
}

/** test sycl::range::get(int index) return size_t
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
      // global size to be set to the size
      sycl::range<1> gs_1d(sizes[0]);
      // local size to be set to 1/4 of the sizes
      sycl::range<1> ls_1d(sizes[0] / 4);
      test_nd_range(log, gs_1d, ls_1d);

      // global size to be set to the size
      sycl::range<2> gs_2d(sizes[0], sizes[1]);
      // local size to be set to 1/4 of the sizes
      sycl::range<2> ls_2d(sizes[0] / 4, sizes[1] / 4);
      test_nd_range(log, gs_2d, ls_2d);

      // global size to be set to the size
      sycl::range<3> gs_3d(sizes[0], sizes[1], sizes[2]);
      // local size to be set to 1/4 of the sizes
      sycl::range<3> ls_3d(sizes[0] / 4, sizes[1] / 4, sizes[2] / 4);
      test_nd_range(log, gs_3d, ls_3d);
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

} /* namespace nd_range_api__ */

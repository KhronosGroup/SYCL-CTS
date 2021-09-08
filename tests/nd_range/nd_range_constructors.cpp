/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME nd_range_constructors

namespace TEST_NAMESPACE {
using namespace sycl_cts;

/**
 * @brief Constructs a default nd_range
 * @tparam dim Number of dimensions of the nd_range
 * @return nd_range<dim> object with default values
 */
template <int dim>
inline sycl::nd_range<dim> get_default_nd_range() {
  const auto range = util::get_cts_object::range<dim>::get(1, 1, 1);
  return sycl::nd_range<dim>(range, range);
}

template <int dim>
void test_nd_range_constructors(util::logger &log, sycl::range<dim> gs,
                                sycl::range<dim> ls) {
  sycl::nd_range<dim> nd_range(gs, ls);

  {  // Copy assignment
    auto defaultRange = get_default_nd_range<dim>();
    defaultRange = nd_range;

    for (int i = 0; i < dim; i++) {
      CHECK_VALUE(log, defaultRange.get_global_range()[i], gs[i], i);
      CHECK_VALUE(log, defaultRange.get_local_range()[i], ls[i], i);
      CHECK_VALUE(log, defaultRange.get_group_range()[i],
                 gs[i] / ls[i], i);
    }
  }
  {  // Move assignment
    auto defaultRange = get_default_nd_range<dim>();
    defaultRange = std::move(nd_range);
    for (int i = 0; i < dim; i++) {
      CHECK_VALUE(log, defaultRange.get_global_range()[i], gs[i], i);
      CHECK_VALUE(log, defaultRange.get_local_range()[i], ls[i], i);
      CHECK_VALUE(log, defaultRange.get_group_range()[i],
                 gs[i] / ls[i], i);
    }
  }
}

/** test sycl::nd_range initialization
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
    constexpr size_t sizes[] = {16, 32, 64};

    try {
      // global size to be set to the size
      sycl::range<1> gs_1d(sizes[0]);
      // local size to be set to 1/4 of the sizes
      sycl::range<1> ls_1d(sizes[0] / 4u);
      test_nd_range_constructors(log, gs_1d, ls_1d);

      // global size to be set to the size
      sycl::range<2> gs_2d(sizes[0], sizes[1]);
      // local size to be set to 1/4 of the sizes
      sycl::range<2> ls_2d(sizes[0] / 4u, sizes[1] / 4u);
      test_nd_range_constructors(log, gs_2d, ls_2d);

      // global size to be set to the size
      sycl::range<3> gs_3d(sizes[0], sizes[1], sizes[2]);
      // local size to be set to 1/4 of the sizes
      sycl::range<3> ls_3d(sizes[0] / 4, sizes[1] / 4, sizes[2] / 4);
      test_nd_range_constructors(log, gs_3d, ls_3d);
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

} /* namespace nd_range_constructors__ */

/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME hierarchical_implicit_barriers

namespace TEST_NAMESPACE {

template <int dim> class kernel;

static const size_t globalItems1d = 8;
static const size_t globalItems2d = 4;
static const size_t globalItems3d = 2;
static const size_t localItems1d = 4;
static const size_t localItems2d = 2;
static const size_t localItems3d = 1;
static const size_t groupRange1d = (globalItems1d / localItems1d);
static const size_t groupRange2d = (globalItems2d / localItems2d);
static const size_t groupItemsTotal =
    (globalItems1d * globalItems2d * globalItems3d);
static const size_t localItemsTotal =
    (localItems1d * localItems2d * localItems3d);
static const size_t groupRangeTotal = (groupItemsTotal / localItemsTotal);

using namespace sycl_cts;

template <int dim> void check_dim(util::logger &log) {
  try {
    size_t inputData[groupItemsTotal];

    auto testQueue = util::get_cts_object::queue();

    for (size_t i = 0; i < groupItemsTotal; i++) {
      inputData[i] = i;
    }
    {
      sycl::buffer<size_t, 1> input_buffer(
          inputData, sycl::range<1>(groupItemsTotal));

      testQueue.submit([&](sycl::handler &cgh) {

        auto globalRange =
            sycl_cts::util::get_cts_object::range<dim>::template get_fixed_size<
                groupRangeTotal>(groupRange1d, groupRange2d);
        auto localRange =
            sycl_cts::util::get_cts_object::range<dim>::template get_fixed_size<
                localItemsTotal>(localItems1d, localItems2d);

        auto inputPtr =
            input_buffer.get_access<sycl::access::mode::read_write>(cgh);

        sycl::accessor<size_t, 1, sycl::access::mode::read_write,
                           sycl::target::local>
            localPtr(sycl::range<1>(localItemsTotal), cgh);

        cgh.parallel_for_work_group<kernel<dim>>(
            globalRange, localRange, [=](sycl::group<dim> group) {
              group.parallel_for_work_item([&](sycl::h_item<dim> item) {
                auto globalId = item.get_global().get_linear_id();
                auto localId = item.get_local().get_linear_id();

                int globalSize = group.get_global_range().size();
                int invertedVal = (globalSize - 1) - inputPtr[globalId];

                localPtr[localId] = invertedVal;
              });

              // Assign inverted val which guaranteed to be already in localPtr
              // due to implicit barrier call
              group.parallel_for_work_item([&](sycl::h_item<dim> item) {
                auto globalId = item.get_global().get_linear_id();
                auto localId = item.get_local().get_linear_id();

                inputPtr[globalId] = localPtr[localId];
              });

            });
      });
    }

    for (size_t i = 0; i < groupItemsTotal; i++) {
      if (inputData[(groupItemsTotal - 1) - i] != i) {
        std::cout << i << " : " << inputData[(groupItemsTotal - 1) - i] << "\n";
        FAIL(log, "Values not equal.");
      }
    }

  } catch (const sycl::exception &e) {
    log_exception(log, e);
    sycl::string_class errorMsg =
        "a SYCL exception was caught: " + sycl::string_class(e.what());
    FAIL(log, errorMsg.c_str());
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

} /* namespace id_api__ */

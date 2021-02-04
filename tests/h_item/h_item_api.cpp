/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/
#include "../common/common.h"

#define TEST_NAME h_item_api

template <int numDims>
struct h_item_api_kernel_common;

/**
 * @brief Kernel struct that tests retrieving a single value from a range or ID
 *        for different ranges and IDs obtained from an h_item
 * @tparam numDims Number of dimensions of the group used in the kernel
 * @tparam currentDim Current dimension to get the value for
 */
template <int numDims, int currentDim>
struct h_item_api_kernel_single {
  cl::sycl::range<numDims> kernelLogicalLocalRange;

  void operator()(cl::sycl::group<numDims> group) const {
    group.parallel_for_work_item(
        kernelLogicalLocalRange, [&](cl::sycl::h_item<numDims> item) {
          size_t globalRange = item.get_global_range(currentDim);
          size_t globalId = item.get_global_id(currentDim);
          size_t localRange = item.get_local_range(currentDim);
          size_t localId = item.get_local_id(currentDim);
          size_t logicalLocalRange = item.get_logical_local_range(currentDim);
          size_t logicalLocalId = item.get_logical_local_id(currentDim);
          size_t physicalLocalRange = item.get_physical_local_range(currentDim);
          size_t physicalLocalId = item.get_physical_local_id(currentDim);
        });
  }
};

namespace TEST_NAME {
using namespace sycl_cts;

/** test cl::sycl::device initialization
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info& out) const final {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  template <int numDims>
  void test_api(util::logger& log) {
    try {
      auto testQueue = util::get_cts_object::queue();

      const auto kernelGroupRange =
          util::get_cts_object::range<numDims>::get(8, 4, 2);
      const auto kernelPhysicalLocalRange =
          util::get_cts_object::range<numDims>::get(4, 2, 1);
      const auto kernelLogicalLocalRange =
          util::get_cts_object::range<numDims>::get(1, 2, 4);

      testQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.parallel_for_work_group<h_item_api_kernel_common<numDims>>(
            kernelGroupRange, kernelPhysicalLocalRange,

            [=](cl::sycl::group<numDims> group) {
              group.parallel_for_work_item(
                  kernelLogicalLocalRange, [&](cl::sycl::h_item<numDims> item) {
                    static constexpr bool with_offset = false;

                    // Get items
                    cl::sycl::item<numDims, with_offset> globalItem =
                        item.get_global();
                    cl::sycl::item<numDims, with_offset> localItem =
                        item.get_local();
                    cl::sycl::item<numDims, with_offset> logicalLocalItem =
                        item.get_logical_local();
                    cl::sycl::item<numDims, with_offset> physicalLocalItem =
                        item.get_physical_local();
                    // Silent warnings
                    (void)globalItem;
                    (void)localItem;
                    (void)logicalLocalItem;
                    (void)physicalLocalItem;

                    // Get ranges
                    cl::sycl::range<numDims> globalRange =
                        item.get_global_range();
                    cl::sycl::range<numDims> localRange =
                        item.get_local_range();
                    cl::sycl::range<numDims> logicalLocalRange =
                        item.get_logical_local_range();
                    cl::sycl::range<numDims> physicalLocalRange =
                        item.get_physical_local_range();
                    // Silent warnings
                    (void)globalRange;
                    (void)localRange;
                    (void)logicalLocalRange;
                    (void)physicalLocalRange;

                    // Get IDs
                    cl::sycl::id<numDims> globalId = item.get_global_id();
                    cl::sycl::id<numDims> localId = item.get_local_id();
                    cl::sycl::id<numDims> logicalLocalId =
                        item.get_logical_local_id();
                    cl::sycl::id<numDims> physicalLocalId =
                        item.get_physical_local_id();
                    // Silent warnings
                    (void)globalId;
                    (void)localId;
                    (void)logicalLocalId;
                    (void)physicalLocalId;
                  });
            });
      });

      if (numDims >= 1) {
        testQueue.submit([&](cl::sycl::handler& cgh) {
          cgh.parallel_for_work_group(
              kernelGroupRange, kernelPhysicalLocalRange,
              h_item_api_kernel_single<numDims, 0>{kernelLogicalLocalRange});
        });
      }
      if (numDims >= 2) {
        testQueue.submit([&](cl::sycl::handler& cgh) {
          cgh.parallel_for_work_group(
              kernelGroupRange, kernelPhysicalLocalRange,
              h_item_api_kernel_single<numDims, 1>{kernelLogicalLocalRange});
        });
      }
      if (numDims >= 3) {
        testQueue.submit([&](cl::sycl::handler& cgh) {
          cgh.parallel_for_work_group(
              kernelGroupRange, kernelPhysicalLocalRange,
              h_item_api_kernel_single<numDims, 2>{kernelLogicalLocalRange});
        });
      }

    } catch (const cl::sycl::exception& e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }

  /** execute the test
   */
  void run(util::logger& log) final {
    test_api<1>(log);
    test_api<2>(log);
    test_api<3>(log);
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAME

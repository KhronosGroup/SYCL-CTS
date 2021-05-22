/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/
#include "../common/common.h"
#include <string>

#define TEST_NAME h_item_api

namespace TEST_NAMESPACE {
using namespace sycl_cts;

template <int numDims>
struct kernel_common;

/**
 * @brief Kernel struct that tests retrieving a single value from a range or ID
 *        for different ranges and IDs obtained from an h_item
 * @tparam numDims Number of dimensions of the group used in the kernel
 * @tparam currentDim Current dimension to get the value for
 */
template <int numDims, int currentDim>
struct kernel_single {
  using success_acc_t =
      cl::sycl::accessor<bool, 1, cl::sycl::access::mode::write>;

  cl::sycl::range<numDims> kernelLogicalLocalRange;
  success_acc_t success_acc;

  void operator()(cl::sycl::group<numDims> group) const {
    group.parallel_for_work_item(
        kernelLogicalLocalRange, [&](cl::sycl::h_item<numDims> item) {
          bool success = true;
          {
            auto value = item.get_global_range(currentDim);
            auto expected = item.get_global_range()[currentDim];
            success &= value == expected;
          }
          {
            auto value = item.get_global_id(currentDim);
            auto expected = item.get_global_id()[currentDim];
            success &= value == expected;
          }
          {
            auto value = item.get_local_range(currentDim);
            auto expected = item.get_local_range()[currentDim];
            success &= value == expected;
          }
          {
            auto value = item.get_local_id(currentDim);
            auto expected = item.get_local_id()[currentDim];
            success &= value == expected;
          }
          {
            auto value = item.get_logical_local_range(currentDim);
            auto expected = item.get_logical_local_range()[currentDim];
            success &= value == expected;
          }
          {
            auto value = item.get_logical_local_id(currentDim);
            auto expected = item.get_logical_local_id()[currentDim];
            success &= value == expected;
          }
          {
            auto value = item.get_physical_local_range(currentDim);
            auto expected = item.get_physical_local_range()[currentDim];
            success &= value == expected;
          }
          {
            auto value = item.get_physical_local_id(currentDim);
            auto expected = item.get_physical_local_id()[currentDim];
            success &= value == expected;
          }

          // Each work-item updates success flag only if it is needed to avoid
          // data race
          if (!success) success_acc[0] = false;
        });
  }
};

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
          util::get_cts_object::range<numDims>::get(3, 4, 2);
      const auto kernelPhysicalLocalRange =
          util::get_cts_object::range<numDims>::get(4, 2, 1);
      const auto kernelLogicalLocalRange =
          util::get_cts_object::range<numDims>::get(8, 4, 3);

      const auto numWorkGroups = kernelGroupRange.size();
      const auto numPhysicalPerGroup = kernelPhysicalLocalRange.size();
      const auto numLogicalPerGroup = kernelLogicalLocalRange.size();
      const auto numPhysicalWorkItems = numWorkGroups * numPhysicalPerGroup;

      bool isConsistent = true;  // stores result of API consistency check

      using dataT = size_t;
      const dataT initialValue = 12345;
      std::vector<dataT> globalIdData(numPhysicalWorkItems, initialValue);
      std::vector<dataT> physicalLocalIdData(numPhysicalPerGroup, initialValue);
      std::vector<dataT> logicalLocalIdData(numLogicalPerGroup, initialValue);

      {
        cl::sycl::buffer<bool> consistency_buf(&isConsistent,
                                               cl::sycl::range<1>(1));

        cl::sycl::buffer<dataT> globalIdBuf(
            globalIdData.data(), cl::sycl::range<1>(numPhysicalWorkItems));
        cl::sycl::buffer<dataT> physicalLocalIdBuf(
            physicalLocalIdData.data(),
            cl::sycl::range<1>(numPhysicalPerGroup));
        cl::sycl::buffer<dataT> logicalLocalIdBuf(
            logicalLocalIdData.data(), cl::sycl::range<1>(numLogicalPerGroup));

        testQueue.submit([&](cl::sycl::handler& cgh) {
          auto consistency_acc =
              consistency_buf.get_access<cl::sycl::access::mode::write>(cgh);

          auto global_acc =
              globalIdBuf.get_access<cl::sycl::access::mode::write>(cgh);
          auto logical_acc =
              logicalLocalIdBuf.get_access<cl::sycl::access::mode::write>(cgh);
          auto physical_acc =
              physicalLocalIdBuf.get_access<cl::sycl::access::mode::write>(cgh);

          cgh.parallel_for_work_group<kernel_common<numDims>>(
              kernelGroupRange, kernelPhysicalLocalRange,

              [=](cl::sycl::group<numDims> group) {
                group.parallel_for_work_item(
                    kernelLogicalLocalRange,
                    [&](cl::sycl::h_item<numDims> item) {
                      bool success = true;
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

                      // Check items
                      success &= localItem == logicalLocalItem;

                      // Store item linear IDs to verify all are present
                      {
                        const size_t globalId = globalItem.get_linear_id();
                        const size_t physicalId =
                            physicalLocalItem.get_linear_id();
                        const size_t logicalId =
                            logicalLocalItem.get_linear_id();

                        if (globalId < numPhysicalWorkItems) {
                          global_acc[globalId] = globalId;
                        } else {
                          success = false;
                        }

                        if (physicalId < numPhysicalPerGroup) {
                          physical_acc[physicalId] = physicalId;
                        } else {
                          success = false;
                        }

                        if (logicalId < numLogicalPerGroup) {
                          logical_acc[logicalId] = logicalId;
                        } else {
                          success = false;
                        }
                      }

                      // Get ranges
                      cl::sycl::range<numDims> globalRange =
                          item.get_global_range();
                      cl::sycl::range<numDims> localRange =
                          item.get_local_range();
                      cl::sycl::range<numDims> logicalLocalRange =
                          item.get_logical_local_range();
                      cl::sycl::range<numDims> physicalLocalRange =
                          item.get_physical_local_range();

                      // Check ranges
                      success &= globalItem.get_range() == globalRange;
                      success &= localItem.get_range() == localRange;
                      success &= logicalLocalItem.get_range() == logicalLocalRange;
                      success &= physicalLocalItem.get_range() == physicalLocalRange;

                      // Get IDs
                      cl::sycl::id<numDims> globalId = item.get_global_id();
                      cl::sycl::id<numDims> localId = item.get_local_id();
                      cl::sycl::id<numDims> logicalLocalId =
                          item.get_logical_local_id();
                      cl::sycl::id<numDims> physicalLocalId =
                          item.get_physical_local_id();

                      // Check IDs
                      success &= globalItem.get_id() == globalId;
                      success &= localItem.get_id() == localId;
                      success &= logicalLocalItem.get_id() == logicalLocalId;
                      success &= physicalLocalItem.get_id() == physicalLocalId;

                      if (!success) consistency_acc[0] = false;
                    });
              });
        });

        if constexpr (numDims >= 1) {
          testQueue.submit([&](cl::sycl::handler& cgh) {
            auto consistency_acc =
                consistency_buf.get_access<cl::sycl::access::mode::write>(cgh);
            kernel_single<numDims, 0> functor{kernelLogicalLocalRange,
                                              consistency_acc};
            cgh.parallel_for_work_group(kernelGroupRange,
                                        kernelPhysicalLocalRange, functor);
          });
        }
        if constexpr (numDims >= 2) {
          testQueue.submit([&](cl::sycl::handler& cgh) {
            auto consistency_acc =
                consistency_buf.get_access<cl::sycl::access::mode::write>(cgh);
            kernel_single<numDims, 1> functor{kernelLogicalLocalRange,
                                              consistency_acc};
            cgh.parallel_for_work_group(kernelGroupRange,
                                        kernelPhysicalLocalRange, functor);
          });
        }
        if constexpr (numDims >= 3) {
          testQueue.submit([&](cl::sycl::handler& cgh) {
            auto consistency_acc =
                consistency_buf.get_access<cl::sycl::access::mode::write>(cgh);
            kernel_single<numDims, 2> functor{kernelLogicalLocalRange,
                                              consistency_acc};
            cgh.parallel_for_work_group(kernelGroupRange,
                                        kernelPhysicalLocalRange, functor);
          });
        }
      }

      // Check h_item API consistency
      if (!isConsistent) {
        FAIL(log, "h_item API consistency checks failed");
      }
      // Check all expected IDs are present
      auto check = [&](size_t value, size_t index, const char* desc) {
        if (value != index) {
          std::string errorMessage(desc);
          errorMessage += " with index ";
          errorMessage += std::to_string(index);

          if (value == initialValue) {
            errorMessage += " was not present";
          } else {
            errorMessage += " has unexpected value: ";
            errorMessage += std::to_string(value);
          }
          FAIL(log, errorMessage);
        }
      };
      for (size_t i = 0; i < numPhysicalWorkItems; ++i) {
        check(globalIdData[i], i, "global id");
      }
      for (size_t i = 0; i < numLogicalPerGroup; ++i) {
        check(logicalLocalIdData[i], i, "logical local id");
      }
      for (size_t i = 0; i < numPhysicalPerGroup; ++i) {
        check(physicalLocalIdData[i], i, "physical local id");
      }
    } catch (const cl::sycl::exception& e) {
      log_exception(log, e);
      auto errorMsg = std::string("a SYCL exception was caught: ") + e.what();
      FAIL(log, errorMsg);
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

}  // namespace TEST_NAMESPACE

/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Common invocation functors for nd_item and group tests
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_COMMON_INVOKE_H
#define __SYCLCTS_TESTS_COMMON_INVOKE_H

#include "../common/common.h"
#include <array>

namespace {

/**
 * @brief Functor to invoke kernels with nd_item in use
 * @tparam dim Dimension to use
 * @tparam kernelT Type to use as the kernel name
  */
template <int dim, typename kernelT>
struct invoke_nd_item {
  static constexpr int dimensions = dim;
  using parameterT = sycl::nd_item<dim>;

  /**
   * @brief Functor body
   * @tparam kernelBodyT Deduced type of the kernel body
   * @param cgh Command group handler to use for invocation
   * @param numWorkItems Global range - total number of work items
   * @param workGroupSize Local range - number of work items per group
   * @param kernelBody Kernel body to call
   */
  template <typename kernelBodyT>
  void operator()(sycl::handler &cgh,
                  sycl::range<dim> numWorkItems,
                  sycl::range<dim> workGroupSize,
                  kernelBodyT kernelBody) {

    cgh.parallel_for<kernelT>(
        sycl::nd_range<dim>(numWorkItems, workGroupSize),
        [=](sycl::nd_item<dim> ndItem) {
          const size_t index = ndItem.get_global_linear_id();

          kernelBody(ndItem, index);
    });
  }
};

/**
 * @brief Functor to invoke kernels with group in use
 * @tparam dim Dimension to use
 * @tparam kernelT Type to use as the kernel name
  */
template <int dim, typename kernelT>
struct invoke_group {
  static constexpr int dimensions = dim;
  using parameterT = sycl::group<dim>;

  /**
   * @brief Functor body
   * @tparam kernelBodyT Deduced type of the kernel body
   * @param cgh Command group handler to use for invocation
   * @param numWorkItems Global range - total number of work items
   * @param workGroupSize Local range - number of work items per group
   * @param kernelBody Kernel body to call
   */
  template <typename kernelBodyT>
  void operator()(sycl::handler &cgh,
                  sycl::range<dim> numWorkItems,
                  sycl::range<dim> workGroupSize,
                  kernelBodyT kernelBody) {
    sycl::range<dim> numWorkGroups = numWorkItems / workGroupSize;

    cgh.parallel_for_work_group<kernelT>(
        numWorkGroups, workGroupSize,
        [=](sycl::group<dim> group) {
            const size_t index = group.get_linear_id();

            kernelBody(group, index);
    });
  }
};

template <int dim, typename kernelT>
struct invoke_sub_group {
  static constexpr int dimensions = dim;
  using parameterT = sycl::sub_group;

template <typename kernelBodyT>
  void operator()(sycl::handler &cgh,
                  sycl::range<dim> numWorkItems,
                  sycl::range<dim> workGroupSize,
                  kernelBodyT kernelBody) {
    sycl::range<dim> numWorkGroups = numWorkItems / workGroupSize;

    cgh.parallel_for<kernelT>(
        sycl::nd_range<dim>(numWorkItems, workGroupSize),
        [=](sycl::nd_item<3> item) {
          const size_t index = item.get_global_linear_id();
          sycl::sub_group sub_group = item.get_sub_group();

          kernelBody(sub_group, index);
    });
  }
};

/**
 * @brief Generate and store the given number of nd_item/group/h_item/sub_group instances
 * @retval Array of instances
 * @tparam numItems Number of instances to store
 * @tparam kernelInvokeT Invocation functor to use
 */
template <size_t numItems, class kernelInvokeT>
std::array<typename kernelInvokeT::parameterT, numItems> store_instances()
{
  constexpr auto numDims = kernelInvokeT::dimensions;
  using item_t = typename kernelInvokeT::parameterT;
  using item_array_t = std::array<item_t, numItems>;
  alignas(alignof(item_array_t)) char rawItems[sizeof(item_array_t)];
  auto& items = reinterpret_cast<item_array_t&>(rawItems);

  const auto oneElemRange =
      sycl_cts::util::get_cts_object::range<numDims>::get(1, 1, 1);
  const auto itemRange =
      sycl_cts::util::get_cts_object::range<numDims>::get(numItems, 1, 1);

  {
    sycl::buffer<item_t> itemBuf(items.data(),
                                     sycl::range<1>(items.size()));

    auto queue = sycl_cts::util::get_cts_object::queue();
    queue.submit([&](sycl::handler& cgh) {
      auto itemAcc =
          itemBuf.template get_access<sycl::access_mode::write>(cgh);

      kernelInvokeT{}(
          cgh, itemRange, oneElemRange,
          [=](item_t& item, const size_t index) {
              itemAcc[index] = item;
      });
    });
    queue.wait_and_throw();
  }
  return items;
}

} // namespace

#endif // __SYCLCTS_TESTS_COMMON_INVOKE_H

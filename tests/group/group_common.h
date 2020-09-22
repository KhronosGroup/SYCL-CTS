/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
// Provides common methods for group tests
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_GROUP_GROUP_COMMON_H
#define __SYCLCTS_TESTS_GROUP_GROUP_COMMON_H

#include "../common/common.h"
#include <array>

namespace {

/** Retrieve group objects and store them
 */
template <typename kernelT, int numDims, size_t numItems>
void store_group_instances(
    std::array<cl::sycl::group<numDims>, numItems>& items)
{
  using item_t = cl::sycl::group<numDims>;

  const auto oneElemRange =
      sycl_cts::util::get_cts_object::range<numDims>::get(1, 1, 1);
  const auto itemRange =
      sycl_cts::util::get_cts_object::range<numDims>::get(numItems, 1, 1);

  cl::sycl::buffer<item_t> itemBuf(items.data(),
                                   cl::sycl::range<1>(items.size()));

  auto queue = sycl_cts::util::get_cts_object::queue();
  queue.submit([&](cl::sycl::handler& cgh) {
    auto itemAcc =
        itemBuf.template get_access<cl::sycl::access::mode::write>(cgh);

    cgh.parallel_for_work_group<kernelT>(
        itemRange, oneElemRange,
        [=](item_t group) { itemAcc[group.get_linear_id()] = group; });
  });
  queue.wait_and_throw();
}

} //namespace

#endif // __SYCLCTS_TESTS_GROUP_GROUP_COMMON_H
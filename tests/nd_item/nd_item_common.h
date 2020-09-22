/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
// Provides common methods for nd_item tests
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_ND_ITEM_ND_ITEM_COMMON_H
#define __SYCLCTS_TESTS_ND_ITEM_ND_ITEM_COMMON_H

#include "../common/common.h"
#include <array>

namespace {

/** Retrieve nd_item objects and store them
 */
template <typename kernelT, int numDims, size_t numItems>
void store_nd_item_instances(
    std::array<cl::sycl::nd_item<numDims>, numItems>& items)
{
  using item_t = cl::sycl::nd_item<numDims>;

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

    cgh.parallel_for<kernelT>(
        cl::sycl::nd_range<numDims>(itemRange, oneElemRange),
        [=](item_t item) {
          itemAcc[item.get_global_linear_id()] = item;
        });
  });
  queue.wait_and_throw();
}

} //namespace

#endif // __SYCLCTS_TESTS_ND_ITEM_ND_ITEM_COMMON_H

/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2020-2023 The Khronos Group Inc.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_COMMON_GET_GROUP_RANGE_H
#define __SYCLCTS_TESTS_COMMON_GET_GROUP_RANGE_H

namespace sycl_cts {
namespace util {
/**
 * Since \p sycl::range by standard provides no default constructor,
 * this function returns a \p sycl::range with 1 for each dimension. */
template <int Dimensions>
sycl::range<Dimensions> get_default_range();

template <>
inline sycl::range<1> get_default_range() {
  return sycl::range<1>{1};
}

template <>
inline sycl::range<2> get_default_range() {
  return sycl::range<2>{1, 1};
}

template <>
inline sycl::range<3> get_default_range() {
  return sycl::range<3>{1, 1, 1};
}

/**
 * @brief Provides range for maximal size work-group
 *        supported by device of a given queue.
 *        Multidimensional work-group range is made
 *        as hypercybic as possible
 * @tparam Dimensions Dimension to use for group instance
 */
template <int Dimensions>
sycl::range<Dimensions> work_group_range(
    sycl::queue queue,
    size_t work_items_limit = std::numeric_limits<size_t>::max()) {
  // query device for work-group sizes
  size_t max_work_item_sizes[Dimensions];
  {
    // FIXME: hipSYCL does not implement
    //        sycl::info::device::max_work_item_sizes<3> property
#if SYCL_CTS_COMPILING_WITH_HIPSYCL
    sycl::id<3> sizes =
        queue.get_device().get_info<sycl::info::device::max_work_item_sizes>();
#else
    sycl::id<3> sizes =
        queue.get_device()
            .get_info<sycl::info::device::max_work_item_sizes<3> >();
#endif
    for (int i = 0; i < Dimensions; ++i) {
      max_work_item_sizes[i] = sizes.get(i);
    }
  }
  size_t max_work_group_size = std::min(
      queue.get_device().get_info<sycl::info::device::max_work_group_size>(),
      work_items_limit);

  // make work-group size as much square/cubic as possible
  size_t work_group_sizes[Dimensions] = {
      std::min(max_work_item_sizes[0], max_work_group_size)};
  if constexpr (Dimensions > 1) {
    size_t rest_work_group_size = max_work_group_size;
    for (int cur_D = Dimensions; cur_D > 1; --cur_D) {
      size_t pref_size = pow(rest_work_group_size, 1. / cur_D) + 1;
      pref_size = std::min(pref_size, max_work_item_sizes[cur_D - 1]);
      // in the worst case of prime rest_work_group_size pref_size comes to 1
      while (rest_work_group_size % pref_size != 0) {
        --pref_size;
      }
      work_group_sizes[cur_D - 1] = pref_size;
      rest_work_group_size /= pref_size;
    }
    work_group_sizes[0] =
        std::min(max_work_item_sizes[0], rest_work_group_size);
  }

  sycl::range<Dimensions> work_group_range = get_default_range<Dimensions>();
  for (int i = 0; i < Dimensions; ++i)
    work_group_range[i] = work_group_sizes[i];

  return work_group_range;
}

/**
 * @brief Provides group size pretty printing
 * @tparam D Dimension of group instance
 */
template <int D>
std::string work_group_print(const sycl::range<D>& work_group_range) {
  std::string res("{ " + std::to_string(work_group_range[0]));
  for (int i = 1; i < D; ++i) res += ", " + std::to_string(work_group_range[i]);
  res += " }";
  return res;
}
}  // namespace util
}  // namespace sycl_cts

#endif  // __SYCLCTS_TESTS_COMMON_GET_GROUP_RANGE_H

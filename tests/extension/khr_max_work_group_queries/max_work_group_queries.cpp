/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2025 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "../../common/common.h"

namespace max_work_group_queries::tests {

TEST_CASE(
    "the max work group queries extension defines the "
    "SYCL_KHR_MAX_WORK_GROUP_QUERIES macro",
    "[khr_max_work_group_queries]") {
#ifdef SYCL_KHR_MAX_WORK_GROUP_QUERIES
  STATIC_REQUIRE(SYCL_KHR_MAX_WORK_GROUP_QUERIES == 1);
#else
  SKIP("SYCL_KHR_MAX_WORK_GROUP_QUERIES is not defined");
#endif
}

#ifdef SYCL_KHR_MAX_WORK_GROUP_QUERIES

template <int DIMENSION>
void check_min_size() {
  auto queue = sycl_cts::util::get_cts_object::queue();
  auto dev = queue.get_device();

  check_get_info_param<sycl::khr::info::device::max_work_group_range<DIMENSION>,
                       sycl::range<DIMENSION>>(dev);

  if (dev.get_info<sycl::info::device::device_type>() ==
      sycl::info::device_type::custom)
    return;
  auto max_work_groups =
      dev.get_info<sycl::khr::info::device::max_work_group_range<DIMENSION>>();

  for (int i = 0; i < DIMENSION; i++) {
    CHECK(max_work_groups[i] >= 1);
  }
}

void check_min_total_size() {
  auto queue = sycl_cts::util::get_cts_object::queue();
  auto dev = queue.get_device();

  check_get_info_param<sycl::khr::info::device::max_work_group_range_size,
                       size_t>(dev);

  if (dev.get_info<sycl::info::device::device_type>() ==
      sycl::info::device_type::custom)
    return;

  auto max_total =
      dev.get_info<sycl::khr::info::device::max_work_group_range_size>();

  CHECK(max_total >= 1);
}

TEST_CASE(
    "if the device is not info::device_type::custom, the minimal size in each "
    "dimension and in total is 1",
    "[khr_max_work_group_queries]") {
  check_min_size<1>();
  check_min_size<2>();
  check_min_size<3>();
  check_min_total_size();
}

#endif  // SYCL_KHR_MAX_WORK_GROUP_QUERIES

}  // namespace max_work_group_queries::tests

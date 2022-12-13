/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2023 The Khronos Group Inc.
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

#include "../common/common.h"

#include <algorithm>
#include <bitset>
#include <limits>
#include <numeric>
#include <sstream>
#include <vector>

/** Each of the functions called by the work-items in the below test. */
enum struct method : int {
  group_id = 0,
  local_id,
  local_range,
  group_range,
  max_local_range,
  group_linear_id,
  local_linear_id,
  group_linear_range,
  local_linear_range,
  leader,
  group_group_linear_id,  // part of group, not part of sub-group like above
  method_count            // defines size, should be last
};

static constexpr auto method_cnt = to_integral(method::method_count);

struct kernel_name;

/** Return a linear index for a global work-item index and a method. */
static size_t get_idx(size_t id, method method) {
  return id * method_cnt + to_integral(method);
}

/**
 * Returns the largest possible size that each dimension of a three dimensional
 * work-group may have such that it is a power of two and
 * smaller or equal to 1024. */
static size_t get_work_group_size(const sycl::device& device) {
  const sycl::id<3> max_work_item_sizes =
      device.get_info<sycl::info::device::max_work_item_sizes<3>>();
  const size_t smallest_max_work_item_size =
      std::min(max_work_item_sizes[0],
               std::min(max_work_item_sizes[1], max_work_item_sizes[2]));
  const size_t desired_work_group_size =
      std::min<size_t>(smallest_max_work_item_size, 1024);
  const size_t max_work_group_size =
      device.get_info<sycl::info::device::max_work_group_size>();

  // calc ilog2(desired_work_group_size), desired_work_group_size is at least 1
  size_t tmp = desired_work_group_size;
  size_t desired_work_group_size_exp = 0;
  while (tmp >>= 1) desired_work_group_size_exp++;

  size_t size = 1;
  for (size_t curr_exp = 1; curr_exp < desired_work_group_size_exp;
       curr_exp++) {
    const size_t curr_size = static_cast<size_t>(1) << curr_exp;
    if (curr_size <= desired_work_group_size &&
        curr_size * curr_size * curr_size <= max_work_group_size) {
      size = curr_size;
    } else {
      break;
    }
  }

  return size;
}

std::string format_range(const sycl::range<3>& range) {
  std::ostringstream ss;
  ss << "(" << range[0] << ", " << range[1] << ", " << range[2] << ")";
  return ss.str();
}

TEST_CASE("sub-group api", "[sub_group]") {
  sycl::device device = sycl_cts::util::get_cts_object::device();
  sycl::queue queue = sycl_cts::util::get_cts_object::queue();
  const sycl::range<3> local_range{2, 2, 2};
  const size_t work_group_count = local_range.size();
  size_t local_size_dim = get_work_group_size(device);
  // if possible, reduce size by one to attempt to create incomplete sub-groups
  if (local_size_dim > 1) {
    --local_size_dim;
  }
  sycl::range<3> local_size{local_size_dim, local_size_dim, local_size_dim};

  const sycl::nd_range<3> execution_range{local_range * local_size, local_size};
  INFO("number of work-groups: " << format_range(local_range)
                                 << " work-group size: "
                                 << format_range(local_size));
  const size_t linear_execution_range =
      execution_range.get_global_range().size();

  // allocate space for results to be gathered,
  // initialize arrays to special value to ensure a write happens
  const size_t result_count = linear_execution_range * method_cnt;
  constexpr size_t special_val = std::numeric_limits<size_t>::max();
  std::vector<size_t> results(result_count, special_val);
  {
    sycl::buffer<size_t, 1> buffer_value(results.data(),
                                         sycl::range<1>(result_count));

    queue.submit([&](sycl::handler& cgh) {
      auto accessor_value =
          buffer_value.get_access<sycl::access_mode::read_write>(cgh);

      cgh.parallel_for<kernel_name>(
          execution_range, [=](sycl::nd_item<3> item) {
            const size_t lgid = item.get_global_linear_id();
            const sycl::sub_group sub_group = item.get_sub_group();

            {  // get_group_id()
              const size_t idx = get_idx(lgid, method::group_id);
              accessor_value[idx] = sub_group.get_group_id()[0];
              ASSERT_RETURN_TYPE(sycl::id<1>, sub_group.get_group_id(),
                                 "sycl::sub_group.get_group_id()");
            }
            {  // get_local_id()
              const size_t idx = get_idx(lgid, method::local_id);
              accessor_value[idx] = sub_group.get_local_id()[0];
              ASSERT_RETURN_TYPE(sycl::id<1>, sub_group.get_local_id(),
                                 "sycl::sub_group.get_local_id()");
            }
            {  // get_local_range()
              const size_t idx = get_idx(lgid, method::local_range);
              accessor_value[idx] = sub_group.get_local_range()[0];
              ASSERT_RETURN_TYPE(sycl::range<1>, sub_group.get_local_range(),
                                 "sycl::sub_group.get_local_range()");
            }
            {  // get_group_range()
              const size_t idx = get_idx(lgid, method::group_range);
              accessor_value[idx] = sub_group.get_group_range()[0];
              ASSERT_RETURN_TYPE(sycl::range<1>, sub_group.get_group_range(),
                                 "sycl::sub_group.get_group_range()");
            }
            {  // get_max_local_range()
              const size_t idx = get_idx(lgid, method::max_local_range);
              accessor_value[idx] = sub_group.get_max_local_range()[0];
              ASSERT_RETURN_TYPE(sycl::range<1>,
                                 sub_group.get_max_local_range(),
                                 "sycl::sub_group.get_max_local_range()");
            }
            {  // get_group_linear_id()
              const size_t idx = get_idx(lgid, method::group_linear_id);
              accessor_value[idx] = sub_group.get_group_linear_id();
              ASSERT_RETURN_TYPE(uint32_t, sub_group.get_group_linear_id(),
                                 "sycl::sub_group.get_group_linear_id()");
            }
            {  // get_local_linear_id()
              const size_t idx = get_idx(lgid, method::local_linear_id);
              accessor_value[idx] = sub_group.get_local_linear_id();
              ASSERT_RETURN_TYPE(uint32_t, sub_group.get_local_linear_id(),
                                 "sycl::sub_group.get_local_linear_id()");
            }
            {  // get_group_linear_range()
              const size_t idx = get_idx(lgid, method::group_linear_range);
              accessor_value[idx] = sub_group.get_group_linear_range();
              ASSERT_RETURN_TYPE(uint32_t, sub_group.get_group_linear_range(),
                                 "sycl::sub_group.get_group_linear_range()");
            }
            {  // get_local_linear_range()
              const size_t idx = get_idx(lgid, method::local_linear_range);
              accessor_value[idx] = sub_group.get_local_linear_range();
              ASSERT_RETURN_TYPE(uint32_t, sub_group.get_local_linear_range(),
                                 "sycl::sub_group.get_local_linear_range()");
            }
            {  // leader()
              const size_t idx = get_idx(lgid, method::leader);
              accessor_value[idx] = sub_group.leader();
              ASSERT_RETURN_TYPE(bool, sub_group.leader(),
                                 "sycl::sub_group.leader()");
            }
            {  // group.get_group_linear_id()
              const size_t idx = get_idx(lgid, method::group_group_linear_id);
              accessor_value[idx] = item.get_group_linear_id();
              // method belongs to sycl::group, no return type check needed
            }
          });
    });
  }

  // check that all elements were assigned a value
  REQUIRE(std::all_of(results.begin(), results.end(),
                      [=](size_t val) { return val != special_val; }));

  // max_local_range of first work-item, used to check if they are all the same
  size_t first_max_local_range = results[get_idx(0, method::max_local_range)];
  // check isolated results for each work-item
  for (size_t i = 0; i < linear_execution_range; i++) {
    INFO("index: " << i);
    const auto get = [=](method method) -> size_t {
      return results[get_idx(i, method)];
    };

    // group_id, group_linear_id
    size_t idx_group_id = get(method::group_id);
    CHECK((idx_group_id == get(method::group_linear_id)));
    CHECK((idx_group_id < get(method::group_linear_range)));

    // local_id, local_linear_id
    size_t idx_local_id = get(method::local_id);
    CHECK((idx_local_id == get(method::local_linear_id)));
    CHECK((idx_local_id < get(method::local_range)));

    // local_range
    size_t idx_local_range = get(method::local_range);
    CHECK((idx_local_range == get(method::local_linear_range)));
    CHECK((idx_local_range <= get(method::max_local_range)));

    // group_range
    CHECK((get(method::group_range) == get(method::group_linear_range)));

    // leader: check that true iff local id is zero
    CHECK((get(method::leader) == (get(method::local_id) == 0)));

    // max_local_range: check that every work-item returns the same value
    CHECK((first_max_local_range == get(method::max_local_range)));
  }

  // max_local_range: check that it is a valid sub-group size
  const std::vector<size_t> valid_sub_group_sizes =
      device.get_info<sycl::info::device::sub_group_sizes>();
  CHECK(
      std::any_of(valid_sub_group_sizes.begin(), valid_sub_group_sizes.end(),
                  [=](size_t size) { return size == first_max_local_range; }));

  // work-items are assigned to sub-groups in an implementation-defined way,
  // and sub_group.get_group_id() cannot be used to identity work-items in the
  // execution range. the below checks verify properties without depending on
  // implementation details. note therefore that "global index" as used below
  // is calculated rather than expected based on the work-item id

  // check that all work-items in the same work-group have the same number
  // of sub-groups. use REQUIRE as sub_group_counts is used by next checks
  // ---------------------------------------------------------------------------

  // for each work-group, the number of sub-groups it contains
  std::vector<size_t> sub_group_counts(work_group_count, special_val);
  // populate and check sub_group_counts by iterating over the work-items
  for (size_t i = 0; i < linear_execution_range; i++) {
    const size_t wg_id = results[get_idx(i, method::group_group_linear_id)];
    const size_t sub_group_count = results[get_idx(i, method::group_range)];
    if (sub_group_counts[wg_id] == special_val) {
      sub_group_counts[wg_id] = sub_group_count;
    } else {
      // if a previous work-item from the same work-group already set the value,
      // check that it is equal
      REQUIRE((sub_group_count == sub_group_counts[wg_id]));
    }
  }
  // check that all elements were assigned a value, meaning that
  // all unique work-groups are processed
  REQUIRE(std::all_of(sub_group_counts.begin(), sub_group_counts.end(),
                      [=](size_t count) { return count != special_val; }));

  // check that each work-group has as many sub-groups as reported previously
  // ---------------------------------------------------------------------------

  // for each work-group, the number of sub-groups before it
  std::vector<size_t> work_group_offsets(work_group_count);
  std::exclusive_scan(sub_group_counts.begin(), sub_group_counts.end(),
                      work_group_offsets.begin(), 0);

  // the number of sub-groups across all work-groups
  size_t sub_group_count =
      std::reduce(sub_group_counts.begin(), sub_group_counts.end(), 0);
  // for each sub-group, whether there exists at least one local item with that
  // sub-group id
  std::vector<bool> sub_group_index_seen(sub_group_count, false);
  // populate sub_group_index_seen
  for (size_t i = 0; i < linear_execution_range; i++) {
    const size_t wg_id = results[get_idx(i, method::group_group_linear_id)];
    const size_t local_sg_id = results[get_idx(i, method::group_id)];
    // #sub-groups before this work-group + sub-group index in this work-group
    //  = global sub-group index
    const size_t global_sg_id = work_group_offsets[wg_id] + local_sg_id;
    sub_group_index_seen[global_sg_id] = true;
  }
  CHECK(std::all_of(sub_group_index_seen.begin(), sub_group_index_seen.end(),
                    [](bool index_is_seen) { return index_is_seen; }));

  // check that all work-items in the same sub-group report the same sub-group
  // size. use REQUIRE as sub_group_sizes is used by next checks
  // ---------------------------------------------------------------------------

  // for each sub-group, the number of work-items it contains
  std::vector<size_t> sub_group_sizes(sub_group_count, special_val);
  // populate and check sub_group_sizes by iterating over the work-items
  for (size_t i = 0; i < linear_execution_range; i++) {
    const size_t wg_id = results[get_idx(i, method::group_group_linear_id)];
    const size_t local_sg_id = results[get_idx(i, method::group_id)];
    // #sub-groups before this work-group + sub-group index in this work-group
    //  = global sub-group index
    const size_t global_sg_id = work_group_offsets[wg_id] + local_sg_id;
    const size_t sub_group_size = results[get_idx(i, method::local_range)];
    if (sub_group_sizes[global_sg_id] == special_val) {
      sub_group_sizes[global_sg_id] = sub_group_size;
    } else {
      // if a previous work-item from the same sub-group already set the value,
      // check that it is equal
      REQUIRE((sub_group_size == sub_group_sizes[global_sg_id]));
    }
  }
  // check that all elements were assigned a value, meaning that
  // all unique sub-groups are processed
  REQUIRE(std::all_of(sub_group_sizes.begin(), sub_group_sizes.end(),
                      [=](size_t size) { return size != special_val; }));

  // check that each sub-group has as many work-items are reported previously
  // ---------------------------------------------------------------------------

  // for each sub-group, the number of work-items before it
  std::vector<size_t> sub_group_offsets(sub_group_count);
  std::exclusive_scan(sub_group_sizes.begin(), sub_group_sizes.end(),
                      sub_group_offsets.begin(), 0);

  // the number of work-items across all sub-groups
  size_t item_count =
      std::reduce(sub_group_sizes.begin(), sub_group_sizes.end(), 0);
  REQUIRE(item_count == linear_execution_range);

  // for each item, whether there exists an item with that item id
  std::vector<bool> item_seen(item_count, false);
  // populate item_seen
  for (size_t i = 0; i < linear_execution_range; i++) {
    const size_t wg_id = results[get_idx(i, method::group_group_linear_id)];
    const size_t local_sg_id = results[get_idx(i, method::group_id)];
    // #sub-groups before this work-group + sub-group index in this work-group
    //  = global sub-group index
    const size_t global_sg_id = work_group_offsets[wg_id] + local_sg_id;
    const size_t item_id = results[get_idx(i, method::local_id)];
    // #items before this sub-group + local item index in this sub-group
    //  = global item index
    const size_t global_item_id = sub_group_offsets[global_sg_id] + item_id;
    item_seen[global_item_id] = true;
  }
  CHECK(std::all_of(item_seen.begin(), item_seen.end(),
                    [](bool index_is_seen) { return index_is_seen; }));
}

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
//  The test executes kernel function on a big number of work groups using
//  almost every work items available for work group on particular device. Test
//  results are checked for all work items in execution range, a buffer is used
//  to keep test results. The test should check available buffer size on device
//  to chose legal work group size.
*******************************************************************************/

#include "../common/common.h"

#include <algorithm>
#include <bitset>
#include <limits>
#include <numeric>
#include <sstream>
#include <vector>

/** Each of the functions called by the work-items in the below test. */
enum struct member_function : int {
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
  member_function_count   // defines size, should be last
};

static constexpr auto member_function_cnt =
    to_integral(member_function::member_function_count);

struct kernel_name;

/** Return a linear index for a global work-item index and a member function. */
static size_t get_idx(size_t id, member_function member_func) {
  return id * member_function_cnt + to_integral(member_func);
}

/**
 * Returns the largest possible sizes that each dimension of a three dimensional
 * work-group may have such that it is a power of two and
 * smaller or equal to 1024. */
static std::vector<size_t> get_3d_work_group_sizes(const sycl::device& device) {
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL
  const sycl::id<3> max_work_item_sizes =
      device.get_info<sycl::info::device::max_work_item_sizes<3>>();
#else
  const sycl::id<3> max_work_item_sizes{1, 1, 1};
  WARN(
      "Implementation does not define device.get_info<max_work_item_sizes>. "
      "Using work-group of size (1, 1, 1).");
#endif
  const size_t max_work_group_size =
      device.get_info<sycl::info::device::max_work_group_size>();

  std::vector<size_t> desired_work_group_sizes = {
      std::min<size_t>(max_work_item_sizes[0], 1024),
      std::min<size_t>(max_work_item_sizes[1], 1024),
      std::min<size_t>(max_work_item_sizes[2], 1024)};

  // calc ilog2(desired_work_group_sizes), desired_work_group_sizes is at least
  // 1
  auto ilog2 = [](size_t dim_size) {
    size_t dim_size_exp = 0;
    while (dim_size >>= 1) dim_size_exp++;
    return dim_size_exp;
  };

  // Initialize sizes of work group dimensions by 1 for first and second
  // dimensions and by 2^n for third dimension, where n is a maximum exponent
  // that ensures that the number of work items in the third dimension is less
  // than the maximum allowed
  std::vector<size_t> actual_work_group_sizes{
      1, 1, size_t{1} << ilog2(desired_work_group_sizes[2])};

  // Lambda multiplies current size of chosen dimension (first or second) of
  // work group by 2 as long as possible Result value of dimension size will be
  // a power-of-two
  auto increase_dim_size = [max_work_group_size](std::vector<size_t>& wg_sizes,
                                                 size_t dim_idx,
                                                 size_t dim_exp) {
    if (dim_idx == 0 || dim_idx == 1) {
      for (size_t curr_exp = 1; curr_exp <= dim_exp; curr_exp++) {
        // Increase by 2 times until total number of work items in the work
        // group exceeds maximum allowed number of work items for chosen device
        // or until dimension size reaches value 2^dim_exp, dim_exp - maximum
        // exponent value
        wg_sizes[dim_idx] <<= 1;
        size_t tot_work_items = wg_sizes[0] * wg_sizes[1] * wg_sizes[2];
        if (tot_work_items > max_work_group_size) {
          wg_sizes[dim_idx] >>= 1;
          break;
        }
      }
    }
  };

  // Increase the size of the second and then first dimension of work group as
  // long as possible as an attempt to get asymmetrical work group range
  increase_dim_size(actual_work_group_sizes, 1,
                    ilog2(desired_work_group_sizes[1]));
  increase_dim_size(actual_work_group_sizes, 0,
                    ilog2(desired_work_group_sizes[0]));

  return actual_work_group_sizes;
}

/** Returns maximum size of buffer keeping test results that can be allocated
    on device. */
inline uint64_t max_device_buf_size() {
  using buf_el_type = size_t;
  sycl::device device = sycl_cts::util::get_cts_object::device();
  uint64_t mem_aloc_size_in_bytes =
      device.get_info<sycl::info::device::max_mem_alloc_size>();
  uint64_t max_buf_size_in_elements =
      mem_aloc_size_in_bytes / sizeof(buf_el_type);
  return max_buf_size_in_elements;
}

/** Returns required size for buffer keeping test results. */
inline size_t result_buf_size(size_t linear_execution_range) {
  return linear_execution_range * member_function_cnt;
}

/** Checks that result buffer can't be allocated on device. */
inline bool result_buffer_cant_be_alloc_on_device(
    size_t requested_buffer_size) {
  return requested_buffer_size > max_device_buf_size();
};

/** Returns global range for given number of work group and work group sizes. */
inline sycl::range<3> make_global_range(
    const sycl::range<3>& work_group_num,
    const std::vector<size_t>& work_group_size_dims) {
  return work_group_num * sycl::range<3>{work_group_size_dims[0],
                                         work_group_size_dims[1],
                                         work_group_size_dims[2]};
}

/**
 * Reduces current size of each dimension of a three dimensional work-group to
 * nearest suitable power-of-two value to ensure test result buffer will be able
 * to be allocated on device. Returns resulting sizes for each dimension. */
std::vector<size_t> reduce_work_group_size_to_suitable_power_of_two(
    const sycl::range<3>& work_group_num,
    std::vector<size_t> current_work_group_size_dims) {
  uint64_t max_dev_buf_size = max_device_buf_size();
  uint64_t requested_res_buffer_size = result_buf_size(
      make_global_range(work_group_num, current_work_group_size_dims).size());
  uint32_t iter = 0;
  // Halve size of each dimension of the work group until result buffer fits
  // into the available memory on the device, dimension index changes at each
  // iteration and takes the value 0, 1 or 2 depending on the iteration number
  // as the remainder of dividing the iteration number by 3
  while (requested_res_buffer_size > max_dev_buf_size) {
    size_t& dim_size = current_work_group_size_dims[iter % 3];
    dim_size = std::max(dim_size >> 1, size_t{1});
    requested_res_buffer_size = result_buf_size(
        make_global_range(work_group_num, current_work_group_size_dims).size());
    iter++;
  }
  return current_work_group_size_dims;
};

std::string format_range(const sycl::range<3>& range) {
  std::ostringstream ss;
  ss << "(" << range[0] << ", " << range[1] << ", " << range[2] << ")";
  return ss.str();
}

/** Convert a returned value to a size_t for storage in array. */
template <typename ReturnType>
size_t to_size_t(ReturnType returned_value) {
  return returned_value;
}

/** sycl::range cannot be implicitly converted to size_t. */
template <>
size_t to_size_t(sycl::range<1> returned_value) {
  return returned_value[0];
}

/**
 * Registers the result of calling a member function.
 * The returned value \p returned_value is converted to \p size_t and stored
 * in the buffer. */
template <typename Accessor, typename ReturnType>
void register_member_function_no_type_check(size_t linear_execution_range,
                                            size_t linear_global_id,
                                            member_function member,
                                            Accessor accessor,
                                            ReturnType returned_value) {
  // prevent writing out of bounds of the result buffer
  if (linear_global_id >= linear_execution_range) {
    return;
  }
  // calculate the unique index of this work-item and member function
  const size_t idx = get_idx(linear_global_id, member);
  // returned value of the function is converted to size_t for uniform storage
  accessor[idx] = to_size_t(returned_value);
}

/**
 * Registers the result of calling a member function. The actual (inferred) type
 * \p ReturnType is checked to be equal to the expected \p ExpectedType.
 * The returned value \p returned_value is converted to \p size_t and stored
 * in the buffer. */
template <typename ExpectedType, typename Accessor, typename ReturnType>
void register_member_function(size_t linear_execution_range,
                              size_t linear_global_id, member_function member,
                              Accessor accessor, ReturnType returned_value) {
  static_assert(std::is_same_v<ExpectedType, ReturnType>);
  register_member_function_no_type_check(linear_execution_range,
                                         linear_global_id, member, accessor,
                                         returned_value);
}

TEST_CASE("sub-group api", "[sub_group]") {
  sycl::device device = sycl_cts::util::get_cts_object::device();
  sycl::queue queue = sycl_cts::util::get_cts_object::queue();
  const sycl::range<3> local_range{2, 3, 5};
  const size_t work_group_count = local_range.size();
  std::vector<size_t> local_size_dims = get_3d_work_group_sizes(device);
  // if possible, reduce size by one to attempt to create incomplete sub-groups
  std::vector<size_t> reduced_local_size_dims{local_size_dims};
  auto reduce_if_possible = [](size_t& dim_size) {
    dim_size = std::max(dim_size - 1, size_t{1});
  };
  std::for_each(reduced_local_size_dims.begin(), reduced_local_size_dims.end(),
                reduce_if_possible);
  // If requested size for result buffer for current work group dimension size
  // exceeds maximum memory allocation size on device, reduce initial
  // local_size_dims to nearest suitable
  // power-of-two value and if possible, again reduce sizes by one to attempt
  // to create incomplete sub-groups
  size_t requested_res_buffer_size = result_buf_size(
      make_global_range(local_range, reduced_local_size_dims).size());
  if (result_buffer_cant_be_alloc_on_device(requested_res_buffer_size)) {
    reduced_local_size_dims = reduce_work_group_size_to_suitable_power_of_two(
        local_range, local_size_dims);
    std::for_each(reduced_local_size_dims.begin(),
                  reduced_local_size_dims.end(), reduce_if_possible);
  }

  sycl::range<3> local_size{reduced_local_size_dims[0],
                            reduced_local_size_dims[1],
                            reduced_local_size_dims[2]};

  const sycl::nd_range<3> execution_range{local_range * local_size, local_size};
  INFO("number of work-groups: " << format_range(local_range)
                                 << " work-group size: "
                                 << format_range(local_size));
  const size_t linear_execution_range =
      execution_range.get_global_range().size();

  // allocate space for results to be gathered,
  // initialize arrays to special value to ensure a write happens
  const size_t result_count = result_buf_size(linear_execution_range);
  constexpr size_t empty = std::numeric_limits<size_t>::max();
  std::vector<size_t> results(result_count, empty);
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

            // get_group_id()
            register_member_function<sycl::id<1>>(
                linear_execution_range, lgid, member_function::group_id,
                accessor_value, sub_group.get_group_id());
            // get_local_id()
            register_member_function<sycl::id<1>>(
                linear_execution_range, lgid, member_function::local_id,
                accessor_value, sub_group.get_local_id());
            // get_local_range()
            register_member_function<sycl::range<1>>(
                linear_execution_range, lgid, member_function::local_range,
                accessor_value, sub_group.get_local_range());
            // get_group_range()
            register_member_function<sycl::range<1>>(
                linear_execution_range, lgid, member_function::group_range,
                accessor_value, sub_group.get_group_range());
            // get_max_local_range()
            register_member_function<sycl::range<1>>(
                linear_execution_range, lgid, member_function::max_local_range,
                accessor_value, sub_group.get_max_local_range());
            // get_group_linear_id()
            register_member_function<uint32_t>(
                linear_execution_range, lgid, member_function::group_linear_id,
                accessor_value, sub_group.get_group_linear_id());
            // get_local_linear_id()
            register_member_function<uint32_t>(
                linear_execution_range, lgid, member_function::local_linear_id,
                accessor_value, sub_group.get_local_linear_id());
            // get_group_linear_range()
            register_member_function<uint32_t>(
                linear_execution_range, lgid,
                member_function::group_linear_range, accessor_value,
                sub_group.get_group_linear_range());
            // get_local_linear_range()
            register_member_function<uint32_t>(
                linear_execution_range, lgid,
                member_function::local_linear_range, accessor_value,
                sub_group.get_local_linear_range());
            // leader()
            register_member_function<bool>(linear_execution_range, lgid,
                                           member_function::leader,
                                           accessor_value, sub_group.leader());
            // group.get_group_linear_id()
            // member function belongs to sycl::group, no type check needed
            register_member_function_no_type_check(
                linear_execution_range, lgid,
                member_function::group_group_linear_id, accessor_value,
                item.get_group_linear_id());
          });
    });
  }

  // check that all elements were assigned a value
  REQUIRE(std::all_of(results.begin(), results.end(),
                      [=](size_t val) { return val != empty; }));

  // helper function to index the results vector
  auto get_res = [&](size_t id, member_function member_func) -> size_t {
    return results[get_idx(id, member_func)];
  };

  // max_local_range of first work-item, used to check if they are all the same
  size_t first_max_local_range = get_res(0, member_function::max_local_range);
  // check isolated results for each work-item
  for (size_t lgid = 0; lgid < linear_execution_range; lgid++) {
    INFO("linear global index: " << lgid);
    auto get = [&](member_function member_func) -> size_t {
      return get_res(lgid, member_func);
    };

    // group_id, group_linear_id:
    // check that the one-dimensional id is equal to the linearized id
    // check that they are within bounds of the number of sub-groups in the wg
    size_t wi_group_id = get(member_function::group_id);
    CHECK((wi_group_id == get(member_function::group_linear_id)));
    CHECK((wi_group_id < get(member_function::group_linear_range)));

    // local_id, local_linear_id:
    // check that the one-dimensional id is equal to the linearized id
    // check that they are within bounds of the size of the sub-group
    size_t wi_local_id = get(member_function::local_id);
    CHECK((wi_local_id == get(member_function::local_linear_id)));
    CHECK((wi_local_id < get(member_function::local_range)));

    // local_range, local_linear_range:
    // check that the one-dimensional range is equal to the linearized range
    // check that they are at most as large as the maximum range in the wg
    size_t wi_local_range = get(member_function::local_range);
    CHECK((wi_local_range == get(member_function::local_linear_range)));
    CHECK((wi_local_range <= get(member_function::max_local_range)));

    // group_range, group_linear_range:
    // check that the one-dimensional range is equal to the linearized range
    CHECK((get(member_function::group_range) ==
           get(member_function::group_linear_range)));

    // leader: check that true iff local id is zero
    CHECK((get(member_function::leader) ==
           (get(member_function::local_id) == 0)));

    // max_local_range: check that every work-item returns the same value
    CHECK((first_max_local_range == get(member_function::max_local_range)));
  }

  // max_local_range: check that it is a valid sub-group size
  const std::vector<size_t> valid_sub_group_sizes =
      device.get_info<sycl::info::device::sub_group_sizes>();
  CHECK(
      std::any_of(valid_sub_group_sizes.begin(), valid_sub_group_sizes.end(),
                  [=](size_t size) { return size == first_max_local_range; }));

  // Work-items are assigned to sub-groups in an implementation-defined way,
  // and sub_group.get_group_id() cannot be used to identity work-items in the
  // execution range. The checks below verify properties without depending on
  // implementation details. Note therefore that "global index" as used below
  // is calculated rather than expected based on the work-item id.

  // Check that all work-items in the same work-group have the same number
  // of sub-groups. Use REQUIRE as sub_group_counts is used by next checks.
  // ---------------------------------------------------------------------------

  // for each work-group, the number of sub-groups it contains
  std::vector<size_t> sub_group_counts(work_group_count, empty);
  // populate and check sub_group_counts by iterating over the work-items
  for (size_t i = 0; i < linear_execution_range; i++) {
    const size_t wg_id = get_res(i, member_function::group_group_linear_id);
    const size_t sub_group_count = get_res(i, member_function::group_range);
    REQUIRE(wg_id < work_group_count);  // bounds check
    if (sub_group_counts[wg_id] == empty) {
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
                      [=](size_t count) { return count != empty; }));

  // Check that each work-group has as many sub-groups as reported previously.
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
    const size_t wg_id = get_res(i, member_function::group_group_linear_id);
    REQUIRE(wg_id < work_group_count);  // bounds check
    const size_t local_sg_id = get_res(i, member_function::group_id);
    // #sub-groups before this work-group + sub-group index in this work-group
    //  = global sub-group index
    const size_t global_sg_id = work_group_offsets[wg_id] + local_sg_id;
    REQUIRE(global_sg_id < sub_group_count);  // bounds check
    sub_group_index_seen[global_sg_id] = true;
  }
  CHECK(std::all_of(sub_group_index_seen.begin(), sub_group_index_seen.end(),
                    [](bool index_is_seen) { return index_is_seen; }));

  // Check that all work-items in the same sub-group report the same sub-group
  // size. Use REQUIRE as sub_group_sizes is used by next checks.
  // ---------------------------------------------------------------------------

  // for each sub-group, the number of work-items it contains
  std::vector<size_t> sub_group_sizes(sub_group_count, empty);
  // populate and check sub_group_sizes by iterating over the work-items
  for (size_t i = 0; i < linear_execution_range; i++) {
    const size_t wg_id = get_res(i, member_function::group_group_linear_id);
    REQUIRE(wg_id < work_group_count);  // bounds check
    const size_t local_sg_id = get_res(i, member_function::group_id);
    // #sub-groups before this work-group + sub-group index in this work-group
    //  = global sub-group index
    const size_t global_sg_id = work_group_offsets[wg_id] + local_sg_id;
    REQUIRE(global_sg_id < sub_group_count);  // bounds check
    const size_t sub_group_size = get_res(i, member_function::local_range);
    if (sub_group_sizes[global_sg_id] == empty) {
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
                      [=](size_t size) { return size != empty; }));

  // Check that each sub-group has as many work-items are reported previously.
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
    const size_t wg_id = get_res(i, member_function::group_group_linear_id);
    REQUIRE(wg_id < work_group_count);  // bounds check
    const size_t local_sg_id = get_res(i, member_function::group_id);
    // #sub-groups before this work-group + sub-group index in this work-group
    //  = global sub-group index
    const size_t global_sg_id = work_group_offsets[wg_id] + local_sg_id;
    REQUIRE(global_sg_id < sub_group_count);  // bounds check
    const size_t item_id = get_res(i, member_function::local_id);
    // #items before this sub-group + local item index in this sub-group
    //  = global item index
    const size_t global_item_id = sub_group_offsets[global_sg_id] + item_id;
    REQUIRE(global_sg_id < item_count);  // bounds check
    item_seen[global_item_id] = true;
  }
  CHECK(std::all_of(item_seen.begin(), item_seen.end(),
                    [](bool index_is_seen) { return index_is_seen; }));
}

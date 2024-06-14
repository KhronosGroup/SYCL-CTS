/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2024 The Khronos Group Inc.
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

#include "../../group_functions/group_functions_common.h"
#include "non_uniform_group_common.h"

template <typename GroupT, typename T>
class broadcast_non_uniform_group_kernel;

/**
 * @brief Provides test for arbitraty non-uniform group broadcast functions
 * @tparam GroupT Type of the non-uniform group to test with
 * @tparam T Type pointed by Ptr
 */
template <typename GroupT, typename T>
void broadcast_non_uniform_group(sycl::queue& queue) {
  const std::string group_name = NonUniformGroupHelper<GroupT>::get_name();

  INFO("Testing broadcast and select for " + group_name);
  if (!NonUniformGroupHelper<GroupT>::is_supported(queue.get_device())) {
    SKIP("Device does not support " + group_name);
  }

  // 4 functions
  constexpr int test_matrix = 4;
  const std::string test_names[test_matrix] = {
      "T group_broadcast(GroupT g, T x)",
      "T group_broadcast(GroupT g, T x, GroupT::linear_id_type "
      "local_linear_id)",
      "T group_broadcast(GroupT g, T x, GroupT::id_type local_id)",
      "T select_from_group(GroupT g, T x, GroupT::id_type local_id)"};

  sycl::range<1> work_group_range = sycl_cts::util::work_group_range<1>(queue);

  for (size_t test_case = 0;
       test_case < NonUniformGroupHelper<GroupT>::num_test_cases; ++test_case) {
    const std::string test_case_name =
        NonUniformGroupHelper<GroupT>::get_test_case_name(test_case);
    INFO("Running test case (" + std::to_string(test_case) + ") with " +
         test_case_name);
    // array to return results
    T origin_values[test_matrix] = {splat_init<T>(0)};
    T broadcasted_values[test_matrix] = {splat_init<T>(0)};
    {
      sycl::buffer<T, 1> origin_values_buf(origin_values,
                                           sycl::range<1>(test_matrix));
      sycl::buffer<T, 1> broadcasted_values_buf(broadcasted_values,
                                                sycl::range<1>(test_matrix));

      queue.submit([&](sycl::handler& cgh) {
        auto origin_values_acc =
            origin_values_buf
                .template get_access<sycl::access::mode::read_write>(cgh);
        auto broadcasted_values_acc =
            broadcasted_values_buf
                .template get_access<sycl::access::mode::read_write>(cgh);

        sycl::nd_range<1> executionRange(work_group_range, work_group_range);
        // Values computed in a kernel depend on global linear id. We need to
        // make sure that there are no overflows
        REQUIRE(executionRange.get_global_range().size() <
                std::numeric_limits<size_t>::max() / 100);

        cgh.parallel_for<broadcast_non_uniform_group_kernel<
            GroupT, T>>(executionRange, [=](sycl::nd_item<1> item) {
          sycl::sub_group sub_group = item.get_sub_group();

          // If this item is not participating in the group, leave early.
          if (!NonUniformGroupHelper<GroupT>::should_participate(sub_group,
                                                                 test_case))
            return;

          GroupT non_uniform_group =
              NonUniformGroupHelper<GroupT>::create(sub_group, test_case);

          // Each work-item computes a unique value
          T value_to_broadcast(splat_init<T>(
              static_cast<size_t>(item.get_global_linear_id() * 100 +
                                  non_uniform_group.get_local_id())));

          // To simplify the test, we are only checking the first group in
          // the first sub-group.
          size_t preferred_group_id =
              NonUniformGroupHelper<GroupT>::preferred_single_worker_group_id(
                  test_case);
          if (item.get_sub_group().get_group_id()[0] == 0 &&
              non_uniform_group.get_group_id()[0] == preferred_group_id) {
            // Find local id of first, last and some third sub-group item in
            // between. Will be used to check different combinations of
            // broadcasting and receiving work-items
            sycl::id<1> first_id = 0;
            sycl::id<1> mid_id = non_uniform_group.get_local_range() / 2;
            sycl::id<1> last_id = non_uniform_group.get_local_range() - 1;

            // Broadcast from the first work-item
            static_assert(
                std::is_same_v<T, decltype(sycl::group_broadcast(
                                      non_uniform_group, value_to_broadcast))>,
                "Return type of group_broadcast(GroupT g, T x) is wrong\n");

            if (non_uniform_group.leader()) {
              // Work-item which does the broadcast stores value to
              // broadcast to use it later as a reference
              origin_values_acc[0] = value_to_broadcast;
            }
            auto broadcasted_value =
                sycl::group_broadcast(non_uniform_group, value_to_broadcast);
            // We read broadcasted value in another work-item
            if (non_uniform_group.get_local_id() == last_id)
              broadcasted_values_acc[0] = broadcasted_value;

            // Broadcast from the last work-item
            static_assert(std::is_same_v<T, decltype(sycl::group_broadcast(
                                                non_uniform_group,
                                                value_to_broadcast, last_id))>,
                          "Return type of group_broadcast(GroupT g, T x, "
                          "GroupT::linear_id_type local_linear_id) is wrong\n");

            if (non_uniform_group.get_local_id() == last_id) {
              // Work-item which does the broadcast stores value to
              // broadcast to use it later as a reference
              origin_values_acc[1] = value_to_broadcast;
            }

            broadcasted_value = sycl::group_broadcast(
                non_uniform_group, value_to_broadcast,
                non_uniform_group.get_local_linear_range() - 1);
            // We read broadcasted value in another work-item
            if (non_uniform_group.get_local_id() == mid_id)
              broadcasted_values_acc[1] = broadcasted_value;

            // Broadcast from a mid work-item
            static_assert(std::is_same_v<T, decltype(sycl::group_broadcast(
                                                non_uniform_group,
                                                value_to_broadcast, mid_id))>,
                          "Return type of group_broadcast(GroupT g, T x, "
                          "GroupT::id_type local_id) is wrong\n");

            if (non_uniform_group.get_local_id() == mid_id) {
              // Work-item which does the broadcast stores value to
              // broadcast to use it later as a reference
              origin_values_acc[2] = value_to_broadcast;
            }
            broadcasted_value = sycl::group_broadcast(
                non_uniform_group, value_to_broadcast, mid_id);
            // We read broadcasted value in another work-item
            if (non_uniform_group.get_local_id() == first_id)
              broadcasted_values_acc[2] = broadcasted_value;

            // Select from the first work-item
            static_assert(std::is_same_v<T, decltype(sycl::select_from_group(
                                                non_uniform_group,
                                                value_to_broadcast, first_id))>,
                          "Return type of select_from_group(GroupT g, T x, "
                          "GroupT::id_type local_id) is wrong\n");

            if (non_uniform_group.get_local_id() == first_id) {
              // Work-item which does the broadcast stores value to
              // broadcast to use it later as a reference
              origin_values_acc[3] = value_to_broadcast;
            }
            broadcasted_value = sycl::select_from_group(
                non_uniform_group, value_to_broadcast, first_id);
            // We read broadcasted value in another work-item
            if (non_uniform_group.get_local_id() == mid_id)
              broadcasted_values_acc[3] = broadcasted_value;
          }
        });
      });
    }
    for (int i = 0; i < test_matrix; ++i) {
      std::string work_group =
          sycl_cts::util::work_group_print(work_group_range);
      CAPTURE(group_name, work_group);
      INFO("Return value of "
           << test_names[i] << " with T = " << type_name<T>() << " is "
           << (equal(broadcasted_values[i], origin_values[i]) ? "right"
                                                              : "wrong"));
      CHECK(equal(broadcasted_values[i], origin_values[i]));
    }
  }
}

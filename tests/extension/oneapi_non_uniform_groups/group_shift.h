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

#include <valarray>

#include "../../group_functions/group_functions_common.h"
#include "non_uniform_group_common.h"

template <typename GroupT, typename T>
class shift_non_uniform_group_kernel;

template <typename GroupT, typename T>
void shift_non_uniform_group(sycl::queue& queue) {
  const std::string group_name = NonUniformGroupHelper<GroupT>::get_name();

  INFO("Testing permute for " + group_name);
  if (!NonUniformGroupHelper<GroupT>::is_supported(queue.get_device())) {
    SKIP("Device does not support " + group_name);
  }

  // 4 functions
  constexpr int test_matrix = 4;
  const std::string test_names[test_matrix] = {
      "T shift_group_left(GroupT g, T x)",
      "T shift_group_left(GroupT g, T x, GroupT::linear_id_type delta)",
      "T shift_group_right(GroupT g, T x)",
      "T shift_group_right(GroupT g, T x, GroupT::linear_id_type delta)"};

  sycl::range<1> work_group_range = sycl_cts::util::work_group_range<1>(queue);
  size_t work_group_size = work_group_range.size();

  for (size_t test_case = 0;
       test_case < NonUniformGroupHelper<GroupT>::num_test_cases; ++test_case) {
    const std::string test_case_name =
        NonUniformGroupHelper<GroupT>::get_test_case_name(test_case);
    INFO("Running test case (" + std::to_string(test_case) + ") with " +
         test_case_name);

    // array to return results:
    std::valarray<bool> res(false, test_matrix * work_group_size);
    {
      sycl::buffer<bool, 1> res_sycl(
          std::begin(res), sycl::range<1>(test_matrix * work_group_size));

      queue.submit([&](sycl::handler& cgh) {
        auto res_acc = res_sycl.get_access<sycl::access::mode::read_write>(cgh);

        sycl::nd_range<1> executionRange(work_group_range, work_group_range);

        cgh.parallel_for<shift_non_uniform_group_kernel<GroupT, T>>(
            executionRange, [=](sycl::nd_item<1> item) {
              sycl::sub_group sub_group = item.get_sub_group();

              // If this item is not participating in the group, they fill their
              // elements in the result with true and leave early.
              if (!NonUniformGroupHelper<GroupT>::should_participate(
                      sub_group, test_case)) {
                res_acc[0 * work_group_size + item.get_local_linear_id()] =
                    true;
                res_acc[1 * work_group_size + item.get_local_linear_id()] =
                    true;
                res_acc[2 * work_group_size + item.get_local_linear_id()] =
                    true;
                res_acc[3 * work_group_size + item.get_local_linear_id()] =
                    true;
                return;
              }

              GroupT non_uniform_group =
                  NonUniformGroupHelper<GroupT>::create(sub_group, test_case);
              const typename GroupT::linear_id_type llid =
                  non_uniform_group.get_local_linear_id();

              T local_var(splat_init<T>(llid + 1));
              T shifted_var(splat_init<T>(llid + 1));

              static_assert(
                  std::is_same_v<T, decltype(sycl::shift_group_left(
                                        non_uniform_group, local_var))>,
                  "Return type of shift_group_left(GroupT g, T x) is wrong\n");

              shifted_var =
                  sycl::shift_group_left(non_uniform_group, local_var);
              res_acc[0 * work_group_size + item.get_local_linear_id()] =
                  equal(shifted_var, splat_init<T>(llid + 2)) ||
                  (llid + 1 >= non_uniform_group.get_local_linear_range());

              static_assert(
                  std::is_same_v<T, decltype(sycl::shift_group_left(
                                        non_uniform_group, local_var, 3))>,
                  "Return type of shift_group_left(GroupT g, T x, "
                  "GroupT::linear_id_type delta) is wrong\n");

              shifted_var =
                  sycl::shift_group_left(non_uniform_group, local_var, 3);
              res_acc[1 * work_group_size + item.get_local_linear_id()] =
                  equal(shifted_var, splat_init<T>(llid + 4)) ||
                  (llid + 3 >= non_uniform_group.get_local_linear_range());

              static_assert(
                  std::is_same_v<T, decltype(sycl::shift_group_right(
                                        non_uniform_group, local_var))>,
                  "Return type of shift_group_right(GroupT g, T x) is wrong\n");

              shifted_var =
                  sycl::shift_group_right(non_uniform_group, local_var);
              res_acc[2 * work_group_size + item.get_local_linear_id()] =
                  equal(shifted_var, splat_init<T>(llid)) || (llid < 1);

              static_assert(
                  std::is_same_v<T, decltype(sycl::shift_group_right(
                                        non_uniform_group, local_var, 2))>,
                  "Return type of shift_group_right(GroupT g, T x, "
                  "GroupT::linear_id_type delta) is wrong\n");

              shifted_var =
                  sycl::shift_group_right(non_uniform_group, local_var, 2);
              res_acc[3 * work_group_size + item.get_local_linear_id()] =
                  equal(shifted_var, splat_init<T>(llid - 1)) || (llid < 2);
            });
      });
    }
    for (int i = 0; i < test_matrix; ++i) {
      bool result = res[i * work_group_size];
      for (size_t j = 1; j < work_group_size; ++j)
        result &= res[i * work_group_size + j];

      std::string work_group =
          sycl_cts::util::work_group_print(work_group_range);
      CAPTURE(group_name, work_group);
      INFO("Value of " << test_names[i] << " with T = " << type_name<T>()
                       << " is " << (result ? "right" : "wrong"));
      CHECK(result);
    }
  }
}

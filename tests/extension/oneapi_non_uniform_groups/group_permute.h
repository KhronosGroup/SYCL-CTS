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
class permute_non_uniform_group_kernel;

template <typename GroupT, typename T>
struct permute_non_uniform_group_test {
  void operator()(sycl::queue& queue) {
    const std::string group_name = NonUniformGroupHelper<GroupT>::get_name();

    INFO("Testing permute for " + group_name);
    if (!NonUniformGroupHelper<GroupT>::is_supported(queue.get_device())) {
      SKIP("Device does not support " + group_name);
    }

    const std::string test_name =
        "T permute_group_by_xor(GroupT g, T x, GroupT::linear_id_type mask)";

    sycl::range<1> work_group_range =
        sycl_cts::util::work_group_range<1>(queue);
    size_t work_group_size = work_group_range.size();

    for (size_t test_case = 0;
         test_case < NonUniformGroupHelper<GroupT>::num_test_cases;
         ++test_case) {
      const std::string test_case_name =
          NonUniformGroupHelper<GroupT>::get_test_case_name(test_case);
      INFO("Running test case (" + std::to_string(test_case) + ") with " +
           test_case_name);

      // array to return results:
      std::valarray<bool> res(false, work_group_size);
      {
        sycl::buffer<bool, 1> res_sycl(std::begin(res),
                                       sycl::range<1>(work_group_size));

        queue.submit([&](sycl::handler& cgh) {
          auto res_acc =
              res_sycl.get_access<sycl::access::mode::read_write>(cgh);

          sycl::nd_range<1> executionRange(work_group_range, work_group_range);

          cgh.parallel_for<permute_non_uniform_group_kernel<GroupT, T>>(
              executionRange, [=](sycl::nd_item<1> item) {
                sycl::sub_group sub_group = item.get_sub_group();

                // If this item is not participating in the group, they fill
                // their elements in the result with true and leave early.
                if (!NonUniformGroupHelper<GroupT>::should_participate(
                        sub_group, test_case)) {
                  res_acc[item.get_local_linear_id()] = true;
                  return;
                }

                GroupT non_uniform_group =
                    NonUniformGroupHelper<GroupT>::create(sub_group, test_case);

                using lin_id_type = typename GroupT::linear_id_type;
                const lin_id_type llid =
                    non_uniform_group.get_local_linear_id();

                T local_var(splat_init<T>(llid + 1));
                T permuted_var(splat_init<T>(llid + 1));

                static_assert(
                    std::is_same_v<T, decltype(sycl::permute_group_by_xor(
                                          non_uniform_group, local_var, 0))>,
                    "Return type of permute_group_by_xor(GroupT g, T x, "
                    "GroupT::linear_id_type mask) is wrong\n");

                bool res = true;
                for (lin_id_type mask = 1u; mask > 0; mask <<= 1) {
                  permuted_var = sycl::permute_group_by_xor(non_uniform_group,
                                                            local_var, mask);
                  res &=
                      equal(permuted_var, splat_init<T>((llid ^ mask) + 1)) ||
                      ((llid ^ mask) >=
                       non_uniform_group.get_local_linear_range());
                }
                res_acc[item.get_local_linear_id()] = res;
              });
        });
      }
      bool result = res[0];
      for (size_t j = 1; j < work_group_size; ++j) result &= res[j];

      std::string work_group =
          sycl_cts::util::work_group_print(work_group_range);
      CAPTURE(group_name, work_group);
      INFO("Value of " << test_name << " with T = " << type_name<T>() << " is "
                       << (result ? "right" : "wrong"));
      CHECK(result);
    }
  }
};

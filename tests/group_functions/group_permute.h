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

#include <valarray>

#include "group_functions_common.h"

template <int D, typename T>
class permute_sub_group_kernel;

template <int D, typename T>
void permute_sub_group(sycl::queue& queue) {
  // 1 function
  constexpr int test_matrix = 1;
  const std::string test_names[test_matrix] = {
      "T permute_group_by_xor(sub_group g, T x, sub_group::linear_id_type "
      "mask)"};

  sycl::range<D> work_group_range = util::work_group_range<D>(queue);
  size_t work_group_size = work_group_range.size();

  // array to return results:
  std::valarray<bool> res(false, test_matrix * work_group_size);
  {
    sycl::buffer<bool, 1> res_sycl(
        std::begin(res), sycl::range<1>(test_matrix * work_group_size));

    queue.submit([&](sycl::handler& cgh) {
      auto res_acc = res_sycl.get_access<sycl::access::mode::read_write>(cgh);

      sycl::nd_range<D> executionRange(work_group_range, work_group_range);

      cgh.parallel_for<permute_sub_group_kernel<D, T>>(
          executionRange, [=](sycl::nd_item<D> item) {
            sycl::sub_group sub_group = item.get_sub_group();
            using lin_id_type = typename sycl::sub_group::linear_id_type;
            const lin_id_type llid = sub_group.get_local_linear_id();

            T local_var(splat_init<T>(llid + 1));
            T permuted_var(splat_init<T>(llid + 1));

            ASSERT_RETURN_TYPE(
                T, sycl::permute_group_by_xor(sub_group, local_var, 0),
                "Return type of permute_group_by_xor(sub_group g, T x, "
                "sub_group::linear_id_type mask) is wrong\n");

            bool res = true;
            for (lin_id_type mask = 1u; mask > 0; mask <<= 1) {
              permuted_var =
                  sycl::permute_group_by_xor(sub_group, local_var, mask);
              res &= equal(permuted_var, splat_init<T>((llid ^ mask) + 1)) ||
                     ((llid ^ mask) >= sub_group.get_local_linear_range());
            }
            res_acc[0 * work_group_size + item.get_local_linear_id()] = res;
          });
    });
  }
  for (int i = 0; i < test_matrix; ++i) {
    bool result = res[i * work_group_size];
    for (size_t j = 1; j < work_group_size; ++j)
      result &= res[i * work_group_size + j];

    std::string work_group = util::work_group_print(work_group_range);
    CAPTURE(D, work_group);
    INFO("Value of " << test_names[i] << " with T = " << type_name<T>()
                     << " is " << (result ? "right" : "wrong"));
    CHECK(result);
  }
}

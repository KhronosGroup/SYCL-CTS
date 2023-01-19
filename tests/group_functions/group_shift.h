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
class shift_group_kernel;

/**
 * @brief Provides test for shift (shuffle) values inside the group
 * @tparam D Dimension to use for group instance
 * @tparam T Type for shifted value
 */
template <int D, typename T>
void shift_group(sycl::queue& queue) {
  // 4 functions
  constexpr int test_matrix = 4;
  const std::string test_names[test_matrix] = {
      "T shift_group_left(group g, T x)",
      "T shift_group_left(group g, T x, group::linear_id_type delta)",
      "T shift_group_right(group g, T x)",
      "T shift_group_right(group g, T x, group::linear_id_type delta)"};

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

      cgh.parallel_for<shift_group_kernel<D, T>>(
          executionRange, [=](sycl::nd_item<D> item) {
            sycl::group<D> group = item.get_group();
            const typename sycl::group<D>::linear_id_type llid =
                item.get_local_linear_id();

            T local_var(init_helper<T>(llid + 1));
            T shifted_var(init_helper<T>(llid + 1));

            ASSERT_RETURN_TYPE(
                T, sycl::shift_group_left(group, local_var),
                "Return type of shift_group_left(group g, T x) is wrong\n");

            shifted_var = sycl::shift_group_left(group, local_var);
            res_acc[0 * work_group_size + llid] =
                equal(shifted_var, init_helper<T>(llid + 2)) ||
                (llid + 1 >= group.get_local_linear_range());

            ASSERT_RETURN_TYPE(T, sycl::shift_group_left(group, local_var, 3),
                               "Return type of shift_group_left(group g, T x, "
                               "group::linear_id_type delta) is wrong\n");

            shifted_var = sycl::shift_group_left(group, local_var, 3);
            res_acc[1 * work_group_size + llid] =
                equal(shifted_var, init_helper<T>(llid + 4)) ||
                (llid + 3 >= group.get_local_linear_range());

            ASSERT_RETURN_TYPE(
                T, sycl::shift_group_right(group, local_var),
                "Return type of shift_group_right(group g, T x) is wrong\n");

            shifted_var = sycl::shift_group_right(group, local_var);
            res_acc[2 * work_group_size + llid] =
                equal(shifted_var, init_helper<T>(llid)) || (llid < 1);

            ASSERT_RETURN_TYPE(T, sycl::shift_group_right(group, local_var, 2),
                               "Return type of shift_group_right(group g, T x, "
                               "group::linear_id_type delta) is wrong\n");

            shifted_var = sycl::shift_group_right(group, local_var, 2);
            res_acc[3 * work_group_size + llid] =
                equal(shifted_var, init_helper<T>(llid - 1)) || (llid < 2);
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

template <int D, typename T>
class shift_sub_group_kernel;

template <int D, typename T>
void shift_sub_group(sycl::queue& queue) {
  // 4 functions
  constexpr int test_matrix = 4;
  const std::string test_names[test_matrix] = {
      "T shift_group_left(sub_group g, T x)",
      "T shift_group_left(sub_group g, T x, sub_group::linear_id_type delta)",
      "T shift_group_right(sub_group g, T x)",
      "T shift_group_right(sub_group g, T x, sub_group::linear_id_type delta)"};

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

      cgh.parallel_for<shift_sub_group_kernel<
          D, T>>(executionRange, [=](sycl::nd_item<D> item) {
        sycl::sub_group sub_group = item.get_sub_group();
        const sycl::sub_group::linear_id_type llid =
            sub_group.get_local_linear_id();

        T local_var(init_helper<T>(llid + 1));
        T shifted_var(init_helper<T>(llid + 1));

        ASSERT_RETURN_TYPE(
            T, sycl::shift_group_left(sub_group, local_var),
            "Return type of shift_group_left(sub_group g, T x) is wrong\n");

        shifted_var = sycl::shift_group_left(sub_group, local_var);
        res_acc[0 * work_group_size + item.get_local_linear_id()] =
            equal(shifted_var, init_helper<T>(llid + 2)) ||
            (llid + 1 >= sub_group.get_local_linear_range());

        ASSERT_RETURN_TYPE(T, sycl::shift_group_left(sub_group, local_var, 3),
                           "Return type of shift_group_left(sub_group g, T x, "
                           "sub_group::linear_id_type delta) is wrong\n");

        shifted_var = sycl::shift_group_left(sub_group, local_var, 3);
        res_acc[1 * work_group_size + item.get_local_linear_id()] =
            equal(shifted_var, init_helper<T>(llid + 4)) ||
            (llid + 3 >= sub_group.get_local_linear_range());

        ASSERT_RETURN_TYPE(
            T, sycl::shift_group_right(sub_group, local_var),
            "Return type of shift_group_right(sub_group g, T x) is wrong\n");

        shifted_var = sycl::shift_group_right(sub_group, local_var);
        res_acc[2 * work_group_size + item.get_local_linear_id()] =
            equal(shifted_var, init_helper<T>(llid)) || (llid < 1);

        ASSERT_RETURN_TYPE(T, sycl::shift_group_right(sub_group, local_var, 2),
                           "Return type of shift_group_right(sub_group g, T x, "
                           "sub_group::linear_id_type delta) is wrong\n");

        shifted_var = sycl::shift_group_right(sub_group, local_var, 2);
        res_acc[3 * work_group_size + item.get_local_linear_id()] =
            equal(shifted_var, init_helper<T>(llid - 1)) || (llid < 2);
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

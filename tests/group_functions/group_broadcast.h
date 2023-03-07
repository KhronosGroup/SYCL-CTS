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

#include "group_functions_common.h"

template <int D, typename T>
class broadcast_group_kernel;

/**
 * @brief Provides test for broadcast values inside the group
 * @tparam D Dimension to use for group instance
 * @tparam T Type for broadcasted value
 */
template <int D, typename T>
void broadcast_group(sycl::queue& queue) {
  // 3 functions
  constexpr int test_matrix = 3;
  const std::string test_names[test_matrix] = {
      "T group_broadcast(group g, T x)",
      "T group_broadcast(group g, T x, group::linear_id_type local_linear_id)",
      "T group_broadcast(group g, T x, group::id_type local_id)"};

  sycl::range<D> work_group_range = sycl_cts::util::work_group_range<D>(queue);
  size_t work_group_size = work_group_range.size();

  // array to return results
  T res[test_matrix] = {splat_init<T>(0)};
  {
    sycl::buffer<T, 1> res_sycl(res, sycl::range<1>(test_matrix));

    queue.submit([&](sycl::handler& cgh) {
      auto res_acc =
          res_sycl.template get_access<sycl::access::mode::read_write>(cgh);

      sycl::nd_range<D> executionRange(work_group_range, work_group_range);

      cgh.parallel_for<broadcast_group_kernel<D, T>>(
          executionRange, [=](sycl::nd_item<D> item) {
            sycl::group<D> group = item.get_group();

            // find local id of last group item
            sycl::id<D> last_item = group.get_local_range();
            for (int i = 0; i < D; ++i) {
              --last_item[i];
            }

            T local_var = splat_init<T>(item.get_local_linear_id() + 1);

            // broadcast from the first workitem
            ASSERT_RETURN_TYPE(
                T, sycl::group_broadcast(group, local_var),
                "Return type of group_broadcast(group g, T x) is wrong\n");

            local_var = sycl::group_broadcast(group, local_var);
            if (item.get_local_linear_id() ==
                group.get_local_linear_range() - 1)
              res_acc[0] = local_var;

            local_var = splat_init<T>(item.get_local_linear_id() + 1);

            // broadcast from the last workitem 1
            ASSERT_RETURN_TYPE(
                T,
                sycl::group_broadcast(group, local_var,
                                      group.get_local_linear_range() - 1),
                "Return type of group_broadcast(group g, T x, "
                "group::linear_id_type local_linear_id) is wrong\n");

            local_var = sycl::group_broadcast(
                group, local_var, group.get_local_linear_range() - 1);
            if (item.get_local_linear_id() == 0) res_acc[1] = local_var;

            local_var = splat_init<T>(item.get_local_linear_id() + 1);

            // broadcast from the last workitem 2
            ASSERT_RETURN_TYPE(
                T, sycl::group_broadcast(group, local_var, last_item),
                "Return type of group_broadcast(group g, T x, group::id_type "
                "local_id) is wrong\n");

            local_var = sycl::group_broadcast(group, local_var, last_item);
            if (item.get_local_linear_id() == 0) res_acc[2] = local_var;
          });
    });
  }

  T expected[test_matrix] = {splat_init<T>(1), splat_init<T>(work_group_size),
                             splat_init<T>(work_group_size)};

  for (int i = 0; i < test_matrix; ++i) {
    std::string work_group = sycl_cts::util::work_group_print(work_group_range);
    CAPTURE(D, work_group);
    INFO("Return value of "
         << test_names[i] << " with T = " << type_name<T>() << " is "
         << (equal(res[i], expected[i]) ? "right" : "wrong"));
    CHECK(equal(res[i], expected[i]));
  }
}

template <int D, typename T>
class broadcast_sub_group_kernel;

template <int D, typename T>
void broadcast_sub_group(sycl::queue& queue) {
  // 4 functions
  constexpr int test_matrix = 4;
  const std::string test_names[test_matrix] = {
      "T group_broadcast(sub_group g, T x)",
      "T group_broadcast(sub_group g, T x, sub_group::linear_id_type "
      "local_linear_id)",
      "T group_broadcast(sub_group g, T x, sub_group::id_type local_id)",
      "T select_from_group(sub_group g, T x, sub_group::id_type local_id)"};

  sycl::range<D> work_group_range = sycl_cts::util::work_group_range<D>(queue);

  // array to return results
  T res[test_matrix + 1] = {splat_init<T>(0)};
  {
    sycl::buffer<T, 1> res_sycl(res, sycl::range<1>(test_matrix + 1));

    queue.submit([&](sycl::handler& cgh) {
      auto res_acc =
          res_sycl.template get_access<sycl::access::mode::read_write>(cgh);

      sycl::nd_range<D> executionRange(work_group_range, work_group_range);

      cgh.parallel_for<broadcast_sub_group_kernel<
          D, T>>(executionRange, [=](sycl::nd_item<D> item) {
        sycl::sub_group sub_group = item.get_sub_group();

        T local_var(splat_init<T>(0));

        if (sub_group.get_group_id()[0] == 0) {
          // find local id of last group item
          sycl::id<1> last_item = sub_group.get_local_range();
          --last_item[0];

          // broadcast from the first workitem
          local_var = splat_init<T>(item.get_global_linear_id() + 1);
          ASSERT_RETURN_TYPE(
              T, sycl::group_broadcast(sub_group, local_var),
              "Return type of group_broadcast(sub_group g, T x) is wrong\n");

          local_var = sycl::group_broadcast(sub_group, local_var);
          if (sub_group.get_local_linear_id() ==
              sub_group.get_local_linear_range() - 1)
            res_acc[0] = local_var;

          // broadcast from the last workitem 1
          local_var = splat_init<T>(item.get_global_linear_id() + 1);
          ASSERT_RETURN_TYPE(
              T, sycl::group_broadcast(sub_group, local_var, last_item),
              "Return type of group_broadcast(sub_group g, T x, "
              "sub_group::linear_id_type local_linear_id) is wrong\n");

          local_var = sycl::group_broadcast(
              sub_group, local_var, sub_group.get_local_linear_range() - 1);
          if (sub_group.get_local_linear_id() == 0) res_acc[1] = local_var;

          // broadcast from the last workitem 2
          local_var = splat_init<T>(item.get_global_linear_id() + 1);
          ASSERT_RETURN_TYPE(
              T, sycl::group_broadcast(sub_group, local_var, last_item),
              "Return type of group_broadcast(sub_group g, T x, "
              "sub_group::id_type local_id) is wrong\n");

          local_var = sycl::group_broadcast(sub_group, local_var, last_item);
          if (sub_group.get_local_linear_id() == 0) res_acc[2] = local_var;

          // select from the last workitem
          local_var = splat_init<T>(item.get_global_linear_id() + 1);
          ASSERT_RETURN_TYPE(
              T, sycl::select_from_group(sub_group, local_var, last_item),
              "Return type of select_from_group(sub_group g, T x, "
              "sub_group::id_type local_id) is wrong\n");

          local_var = sycl::select_from_group(sub_group, local_var, last_item);
          if (sub_group.get_local_linear_id() == 0) res_acc[3] = local_var;

          // return the sub-group size
          if (sub_group.get_local_linear_id() == 0)
            res_acc[4] = sub_group.get_local_linear_range();
        }
      });
    });
  }
  T expected[test_matrix] = {splat_init<T>(1), res[4], res[4], res[4]};
  for (int i = 0; i < test_matrix; ++i) {
    std::string work_group = sycl_cts::util::work_group_print(work_group_range);
    CAPTURE(D, work_group);
    INFO("Return value of "
         << test_names[i] << " with T = " << type_name<T>() << " is "
         << (equal(res[i], expected[i]) ? "right" : "wrong"));
    CHECK(equal(res[i], expected[i]));
  }
}

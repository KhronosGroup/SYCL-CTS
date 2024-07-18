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
class joint_of_group_kernel;

/**
 * @brief Provides test for joint non-uniform group bool of operations with
 * predicate functions
 * @tparam GroupT Type of the non-uniform group to test with
 * @tparam T Type pointed by Ptr
 */
template <typename GroupT, typename T>
struct joint_of_group_test {
  void operator()(const std::string& type_name) {
    auto queue = once_per_unit::get_queue();

    const std::string group_name = NonUniformGroupHelper<GroupT>::get_name();

    INFO("Testing group-of predicate function for " + group_name);
    if (!NonUniformGroupHelper<GroupT>::is_supported(queue.get_device())) {
      SKIP("Device does not support " + group_name);
    }

    // 3 functions * 4 predicates
    constexpr int test_matrix = 3;
    const std::string test_names[test_matrix] = {
        "bool joint_any_of(GroupT g, Ptr first, Ptr last, Predicate pred)",
        "bool joint_all_of(GroupT g, Ptr first, Ptr last, Predicate pred)",
        "bool joint_none_of(GroupT g, Ptr first, Ptr last, Predicate pred)"};
    constexpr int test_cases = 4;
    const std::string test_cases_names[test_cases] = {"none true", "one true",
                                                      "some true", "all true"};

    sycl::range<1> work_group_range =
        sycl_cts::util::work_group_range<1>(queue);
    size_t work_group_size = work_group_range.size();

    const size_t sizes[3] = {5, work_group_size / 2, 3 * work_group_size};
    for (size_t test_case = 0;
         test_case < NonUniformGroupHelper<GroupT>::num_test_cases;
         ++test_case) {
      const std::string test_case_name =
          NonUniformGroupHelper<GroupT>::get_test_case_name(test_case);
      INFO("Running test case (" + std::to_string(test_case) + ") with " +
           test_case_name);

      for (size_t size : sizes) {
        std::vector<T> v(size);
        std::iota(v.begin(), v.end(), 1);

        sycl::buffer<T, 1> v_sycl(v.data(), sycl::range<1>(size));

        sycl::buffer<bool, 2> res_sycl(
            sycl::range<2>(work_group_size, test_matrix * test_cases));

        queue.submit([&](sycl::handler& cgh) {
          auto v_acc =
              v_sycl.template get_access<sycl::access::mode::read_write>(cgh);
          auto res_acc =
              res_sycl.get_access<sycl::access::mode::read_write>(cgh);

          sycl::nd_range<1> executionRange(work_group_range, work_group_range);

          cgh.parallel_for<joint_of_group_kernel<
              GroupT, T>>(executionRange, [=](sycl::nd_item<1> item) {
            size_t gid = item.get_global_linear_id();
            sycl::sub_group sub_group = item.get_sub_group();

            // If this item is not participating in the group, leave early.
            if (!NonUniformGroupHelper<GroupT>::should_participate(sub_group,
                                                                   test_case)) {
              // If an item is not participating, its results are trivially
              // correct.
              for (unsigned i = 0; i < test_matrix * test_cases; ++i)
                res_acc[gid][i] = true;
              return;
            }

            GroupT non_uniform_group =
                NonUniformGroupHelper<GroupT>::create(sub_group, test_case);

            T* v_begin = v_acc.get_pointer();
            T* v_end = v_begin + v_acc.size();

            // predicates
            auto none_true = [&](T i) { return i == 0; };
            auto one_true = [&](T i) { return i == 1; };
            auto some_true = [&](T i) { return i > size / 2; };
            auto all_true = [&](T i) { return i <= size; };

            static_assert(
                std::is_same_v<bool, decltype(sycl::joint_any_of(
                                         non_uniform_group, v_begin, v_end,
                                         none_true))>,
                "Return type of joint_any_of(GroupT g, Ptr first, Ptr last, "
                "Predicate pred) is wrong\n");
            res_acc[gid][0] = !sycl::joint_any_of(non_uniform_group, v_begin,
                                                  v_end, none_true);
            res_acc[gid][1] =
                sycl::joint_any_of(non_uniform_group, v_begin, v_end, one_true);
            res_acc[gid][2] = sycl::joint_any_of(non_uniform_group, v_begin,
                                                 v_end, some_true);
            res_acc[gid][3] =
                sycl::joint_any_of(non_uniform_group, v_begin, v_end, all_true);

            static_assert(
                std::is_same_v<bool, decltype(sycl::joint_all_of(
                                         non_uniform_group, v_begin, v_end,
                                         none_true))>,
                "Return type of joint_all_of(GroupT g, Ptr first, Ptr last, "
                "Predicate pred) is wrong\n");
            res_acc[gid][4] = !sycl::joint_all_of(non_uniform_group, v_begin,
                                                  v_end, none_true);
            res_acc[gid][5] = !sycl::joint_all_of(non_uniform_group, v_begin,
                                                  v_end, one_true);
            res_acc[gid][6] = !sycl::joint_all_of(non_uniform_group, v_begin,
                                                  v_end, some_true);
            res_acc[gid][7] =
                sycl::joint_all_of(non_uniform_group, v_begin, v_end, all_true);

            static_assert(
                std::is_same_v<bool, decltype(sycl::joint_none_of(
                                         non_uniform_group, v_begin, v_end,
                                         none_true))>,
                "Return type of joint_none_of(GroupT g, Ptr first, Ptr last, "
                "Predicate pred) is wrong\n");
            res_acc[gid][8] = sycl::joint_none_of(non_uniform_group, v_begin,
                                                  v_end, none_true);
            res_acc[gid][9] = !sycl::joint_none_of(non_uniform_group, v_begin,
                                                   v_end, one_true);
            res_acc[gid][10] = !sycl::joint_none_of(non_uniform_group, v_begin,
                                                    v_end, some_true);
            res_acc[gid][11] = !sycl::joint_none_of(non_uniform_group, v_begin,
                                                    v_end, all_true);
          });
        });
        {
          sycl::host_accessor res_host{res_sycl};
          for (size_t gid = 0; gid < work_group_size; ++gid) {
            int index = 0;
            for (int i = 0; i < test_matrix; ++i)
              for (int j = 0; j < test_cases; ++j) {
                std::string work_group =
                    sycl_cts::util::work_group_print(work_group_range);
                CAPTURE(group_name, work_group);
                INFO("Value of " << test_names[i] << " with "
                                 << test_cases_names[j] << " for item " << gid
                                 << " predicate is "
                                 << (res_host[gid][index] ? "right" : "wrong"));
                CHECK(res_host[gid][index++]);
              }
          }
        }
      }
    }
  }
};

template <typename GroupT, typename T>
class predicate_function_of_non_uniform_group_kernel;

/**
 * @brief Provides test for arbitraty non-uniform group bool of operations with
 * predicate functions
 * @tparam GroupT Type of the non-uniform group to test with
 * @tparam T Type pointed by Ptr
 */
template <typename GroupT, typename T>
struct predicate_function_of_non_uniform_group_test {
  void operator()(const std::string& type_name) {
    auto queue = once_per_unit::get_queue();

    const std::string group_name = NonUniformGroupHelper<GroupT>::get_name();

    // 3 functions * 4 predicates
    constexpr int test_matrix = 3;
    const std::string test_names[test_matrix] = {
        "bool any_of_group(GroupT g, T x, Predicate pred)",
        "bool all_of_group(GroupT g, T x, Predicate pred)",
        "bool none_of_group(GroupT g, T x, Predicate pred)"};
    constexpr int test_cases = 4;
    const std::string test_cases_names[test_cases] = {"none true", "one true",
                                                      "some true", "all true"};

    sycl::range<1> work_group_range =
        sycl_cts::util::work_group_range<1>(queue);

    for (size_t test_case = 0;
         test_case < NonUniformGroupHelper<GroupT>::num_test_cases;
         ++test_case) {
      const std::string test_case_name =
          NonUniformGroupHelper<GroupT>::get_test_case_name(test_case);
      INFO("Running test case (" + std::to_string(test_case) + ") with " +
           test_case_name);

      // test cases: 4 predicates * 3 functions
      constexpr int total_case_count = test_matrix * test_cases;
      sycl::buffer<bool, 2> res_sycl(
          sycl::range<2>(work_group_range.size(), total_case_count));

      queue.submit([&](sycl::handler& cgh) {
        auto res_acc = res_sycl.get_access<sycl::access::mode::read_write>(cgh);

        sycl::nd_range<1> executionRange(work_group_range, work_group_range);

        cgh.parallel_for<predicate_function_of_non_uniform_group_kernel<
            GroupT, T>>(executionRange, [=](sycl::nd_item<1> item) {
          size_t gid = item.get_global_linear_id();
          sycl::sub_group sub_group = item.get_sub_group();

          // If this item is not participating in the group, leave early.
          if (!NonUniformGroupHelper<GroupT>::should_participate(sub_group,
                                                                 test_case))
            return;

          GroupT non_uniform_group =
              NonUniformGroupHelper<GroupT>::create(sub_group, test_case);

          size_t size = non_uniform_group.get_local_linear_range();

          // Use the non-uniform group local ID (plus 1) as a variable against
          // which to test our predicates. Note that this has a well-defined set
          // of values [1,2,...,N] where N is the non-uniform group size. Note
          // that the non-uniform group could also just be of size 1.
          T local_var(non_uniform_group.get_local_linear_id() + 1);

          // predicates
          // The variable is never 1 for any member of the non-uniform group
          auto none_true = [&](T i) { return i == 0; };
          // Exactly one member of the non-uniform group has value 1 (the first)
          auto one_true = [&](T i) { return i == 1; };
          // Some (or all, for non-uniform groups of size 1) members of the
          // non-uniform group have this value
          auto some_true = [&](T i) { return i > size / 2; };
          // The variable is less than or equal to the non-uniform group size
          // for all members of the non-uniform group.
          auto all_true = [&](T i) { return i <= size; };

          {
            static_assert(
                std::is_same_v<bool,
                               decltype(sycl::any_of_group(
                                   non_uniform_group, local_var, none_true))>,
                "Return type of any_of_group(GroupT g, bool pred) is wrong\n");
            res_acc[gid][0] =
                !sycl::any_of_group(non_uniform_group, local_var, none_true);
            res_acc[gid][1] =
                sycl::any_of_group(non_uniform_group, local_var, one_true);
            res_acc[gid][2] =
                sycl::any_of_group(non_uniform_group, local_var, some_true);
            res_acc[gid][3] =
                sycl::any_of_group(non_uniform_group, local_var, all_true);

            static_assert(
                std::is_same_v<bool,
                               decltype(sycl::all_of_group(
                                   non_uniform_group, local_var, none_true))>,
                "Return type of all_of_group(GroupT g, bool pred) is wrong\n");
            res_acc[gid][4] =
                !sycl::all_of_group(non_uniform_group, local_var, none_true);
            // Note that 'one_true' returns true for the first item. Thus in the
            // case that the non-uniform group size is 1, check that all items
            // match; otherwise check that not all items match.
            res_acc[gid][5] =
                sycl::all_of_group(non_uniform_group, local_var, one_true) ^
                (size != 1);
            // Note that 'some_true' returns true for the first item if the
            // non-uniform group size is 1. In that case, check that all items
            // match; otherwise check that not all items match.
            res_acc[gid][6] =
                sycl::all_of_group(non_uniform_group, local_var, some_true) ^
                (size != 1);
            res_acc[gid][7] =
                sycl::all_of_group(non_uniform_group, local_var, all_true);

            static_assert(
                std::is_same_v<bool,
                               decltype(sycl::none_of_group(
                                   non_uniform_group, local_var, none_true))>,
                "Return type of none_of_group(GroupT g, bool pred) is "
                "wrong\n");
            res_acc[gid][8] =
                sycl::none_of_group(non_uniform_group, local_var, none_true);
            res_acc[gid][9] =
                !sycl::none_of_group(non_uniform_group, local_var, one_true);
            res_acc[gid][10] =
                !sycl::none_of_group(non_uniform_group, local_var, some_true);
            res_acc[gid][11] =
                !sycl::none_of_group(non_uniform_group, local_var, all_true);
          }
        });
      });

      {
        sycl::host_accessor res_host{res_sycl};
        for (size_t gid = 0; gid < work_group_range.size(); ++gid) {
          int index = 0;
          for (int i = 0; i < test_matrix; ++i)
            for (int j = 0; j < test_cases; ++j) {
              std::string work_group =
                  sycl_cts::util::work_group_print(work_group_range);
              CAPTURE(group_name, work_group);
              INFO("Value of " << test_names[i] << " with "
                               << test_cases_names[j] << " for item " << gid
                               << " predicate is "
                               << (res_host[gid][index] ? "right" : "wrong"));
              CHECK(res_host[gid][index++]);
            }
        }
      }
    }
  }
};

template <typename GroupT>
class predicate_function_of_non_uniform_group_bool_kernel;

/**
 * @brief Provides test for group bool of operations
 * @tparam GroupT Type of the non-uniform group to test with
 */
template <typename GroupT>
struct bool_function_of_non_uniform_group_test {
  void operator()() {
    auto queue = once_per_unit::get_queue();
    const std::string group_name = NonUniformGroupHelper<GroupT>::get_name();

    INFO("Testing group-of bool function for " + group_name);
    if (!NonUniformGroupHelper<GroupT>::is_supported(queue.get_device())) {
      SKIP("Device does not support " + group_name);
    }

    // 3 functions * 4 predicates
    constexpr int test_matrix = 3;
    const std::string test_names[test_matrix] = {
        "bool any_of_group(GroupT g, bool pred)",
        "bool all_of_group(GroupT g, bool pred)",
        "bool none_of_group(GroupT g, bool pred)"};
    constexpr int test_cases = 4;
    const std::string test_cases_names[test_cases] = {"none true", "one true",
                                                      "some true", "all true"};

    using T = size_t;

    sycl::range<1> work_group_range =
        sycl_cts::util::work_group_range<1>(queue);

    for (size_t test_case = 0;
         test_case < NonUniformGroupHelper<GroupT>::num_test_cases;
         ++test_case) {
      const std::string test_case_name =
          NonUniformGroupHelper<GroupT>::get_test_case_name(test_case);
      INFO("Running test case (" + std::to_string(test_case) + ") with " +
           test_case_name);

      // test cases: 4 predicates * 3 functions
      constexpr int total_case_count = test_matrix * test_cases;
      sycl::buffer<bool, 2> res_sycl(
          sycl::range<2>(work_group_range.size(), total_case_count));

      queue.submit([&](sycl::handler& cgh) {
        auto res_acc = res_sycl.get_access<sycl::access::mode::read_write>(cgh);

        sycl::nd_range<1> executionRange(work_group_range, work_group_range);

        cgh.parallel_for<predicate_function_of_non_uniform_group_bool_kernel<
            GroupT>>(executionRange, [=](sycl::nd_item<1> item) {
          size_t gid = item.get_global_linear_id();
          sycl::sub_group sub_group = item.get_sub_group();

          // If this item is not participating in the group, leave early.
          if (!NonUniformGroupHelper<GroupT>::should_participate(sub_group,
                                                                 test_case))
            return;

          GroupT non_uniform_group =
              NonUniformGroupHelper<GroupT>::create(sub_group, test_case);

          size_t size = non_uniform_group.get_local_linear_range();

          // Use the non-uniform group local ID (plus 1) as a variable against
          // which to test our predicates. Note that this has a well-defined set
          // of values [1,2,...,N] where N is the non-uniform group size. Note
          // that the non-uniform group could also just be of size 1.
          T local_var(non_uniform_group.get_local_linear_id() + 1);

          // predicates
          // The variable is never 1 for any member of the non-uniform group
          auto none_true = [&](T i) { return i == 0; };
          // Exactly one member of the non-uniform group has value 1 (the first)
          auto one_true = [&](T i) { return i == 1; };
          // Some (or all, for non-uniform groups of size 1) members of the
          // non-uniform group have this value
          auto some_true = [&](T i) { return i > size / 2; };
          // The variable is less than or equal to the non-uniform group size
          // for all members of the non-uniform group.
          auto all_true = [&](T i) { return i <= size; };

          {
            static_assert(
                std::is_same_v<bool,
                               decltype(sycl::any_of_group(
                                   non_uniform_group, none_true(local_var)))>,
                "Return type of any_of_group(GroupT g, bool pred) is wrong\n");
            res_acc[gid][0] =
                !sycl::any_of_group(non_uniform_group, none_true(local_var));
            res_acc[gid][1] =
                sycl::any_of_group(non_uniform_group, one_true(local_var));
            res_acc[gid][2] =
                sycl::any_of_group(non_uniform_group, some_true(local_var));
            res_acc[gid][3] =
                sycl::any_of_group(non_uniform_group, all_true(local_var));

            static_assert(
                std::is_same_v<bool,
                               decltype(sycl::all_of_group(
                                   non_uniform_group, none_true(local_var)))>,
                "Return type of all_of_group(GroupT g, bool pred) is wrong\n");
            res_acc[gid][4] =
                !sycl::all_of_group(non_uniform_group, none_true(local_var));
            // Note that 'one_true' returns true for the first item. Thus in the
            // case that the non-uniform group size is 1, check that all items
            // match; otherwise check that not all items match.
            res_acc[gid][5] =
                sycl::all_of_group(non_uniform_group, one_true(local_var)) ^
                (size != 1);
            // Note that 'some_true' returns true for the first item if the
            // non-uniform group size is 1. In that case, check that all items
            // match; otherwise check that not all items match.
            res_acc[gid][6] =
                sycl::all_of_group(non_uniform_group, some_true(local_var)) ^
                (size != 1);
            res_acc[gid][7] =
                sycl::all_of_group(non_uniform_group, all_true(local_var));

            static_assert(
                std::is_same_v<bool,
                               decltype(sycl::none_of_group(
                                   non_uniform_group, none_true(local_var)))>,
                "Return type of none_of_group(GroupT g, bool pred) is "
                "wrong\n");
            res_acc[gid][8] =
                sycl::none_of_group(non_uniform_group, none_true(local_var));
            res_acc[gid][9] =
                !sycl::none_of_group(non_uniform_group, one_true(local_var));
            res_acc[gid][10] =
                !sycl::none_of_group(non_uniform_group, some_true(local_var));
            res_acc[gid][11] =
                !sycl::none_of_group(non_uniform_group, all_true(local_var));
          }
        });
      });

      {
        sycl::host_accessor res_host{res_sycl};
        for (size_t gid = 0; gid < work_group_range.size(); ++gid) {
          int index = 0;
          for (int i = 0; i < test_matrix; ++i)
            for (int j = 0; j < test_cases; ++j) {
              std::string work_group =
                  sycl_cts::util::work_group_print(work_group_range);
              CAPTURE(group_name, work_group);
              INFO("Value of " << test_names[i] << " with "
                               << test_cases_names[j] << " for item " << gid
                               << " predicate is "
                               << (res_host[gid][index] ? "right" : "wrong"));
              CHECK(res_host[gid][index++]);
            }
        }
      }
    }
  }
};

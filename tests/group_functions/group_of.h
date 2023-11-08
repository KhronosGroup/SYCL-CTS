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
class joint_of_group_kernel;

/**
 * @brief Provides test for joint group bool of operations with predicate
 * functions
 * @tparam D Dimension to use for group instance
 * @tparam T Type pointed by Ptr
 */
template <int D, typename T>
void joint_of_group(sycl::queue& queue) {
  // 6 functions * 4 predicates
  constexpr int test_matrix = 6;
  const std::string test_names[test_matrix] = {
      "bool joint_any_of(group g, Ptr first, Ptr last, Predicate pred)",
      "bool joint_all_of(group g, Ptr first, Ptr last, Predicate pred)",
      "bool joint_none_of(group g, Ptr first, Ptr last, Predicate pred)",
      "bool joint_any_of(sub_group g, Ptr first, Ptr last, Predicate pred)",
      "bool joint_all_of(sub_group g, Ptr first, Ptr last, Predicate pred)",
      "bool joint_none_of(sub_group g, Ptr first, Ptr last, Predicate pred)"};
  constexpr int test_cases = 4;
  const std::string test_cases_names[test_cases] = {"none true", "one true",
                                                    "some true", "all true"};

  sycl::range<D> work_group_range = sycl_cts::util::work_group_range<D>(queue);
  size_t work_group_size = work_group_range.size();

  const size_t sizes[3] = {5, work_group_size / 2, 3 * work_group_size};
  for (size_t size : sizes) {
    std::vector<T> v(size);
    std::iota(v.begin(), v.end(), 1);

    // array to return results:
    bool res[test_matrix * test_cases] = {false};
    {
      sycl::buffer<T, 1> v_sycl(v.data(), sycl::range<1>(size));

      sycl::buffer<bool, 1> res_sycl(res,
                                     sycl::range<1>(test_matrix * test_cases));

      queue.submit([&](sycl::handler& cgh) {
        auto v_acc =
            v_sycl.template get_access<sycl::access::mode::read_write>(cgh);
        auto res_acc = res_sycl.get_access<sycl::access::mode::read_write>(cgh);

        sycl::nd_range<D> executionRange(work_group_range, work_group_range);

        cgh.parallel_for<joint_of_group_kernel<D, T>>(
            executionRange, [=](sycl::nd_item<D> item) {
              T* v_begin = v_acc.get_pointer();
              T* v_end = v_begin + v_acc.size();

              // predicates
              auto none_true = [&](T i) { return i == 0; };
              auto one_true = [&](T i) { return i == 1; };
              auto some_true = [&](T i) { return i > size / 2; };
              auto all_true = [&](T i) { return i <= size; };

              sycl::group<D> group = item.get_group();

              ASSERT_RETURN_TYPE(
                  bool, sycl::joint_any_of(group, v_begin, v_end, none_true),
                  "Return type of joint_any_of(Group g, Ptr first, Ptr last, "
                  "Predicate pred) is wrong\n");
              res_acc[0] =
                  !sycl::joint_any_of(group, v_begin, v_end, none_true);
              res_acc[1] = sycl::joint_any_of(group, v_begin, v_end, one_true);
              res_acc[2] = sycl::joint_any_of(group, v_begin, v_end, some_true);
              res_acc[3] = sycl::joint_any_of(group, v_begin, v_end, all_true);

              ASSERT_RETURN_TYPE(
                  bool, sycl::joint_all_of(group, v_begin, v_end, none_true),
                  "Return type of joint_all_of(Group g, Ptr first, Ptr last, "
                  "Predicate pred) is wrong\n");
              res_acc[4] =
                  !sycl::joint_all_of(group, v_begin, v_end, none_true);
              res_acc[5] = !sycl::joint_all_of(group, v_begin, v_end, one_true);
              res_acc[6] =
                  !sycl::joint_all_of(group, v_begin, v_end, some_true);
              res_acc[7] = sycl::joint_all_of(group, v_begin, v_end, all_true);

              ASSERT_RETURN_TYPE(
                  bool, sycl::joint_none_of(group, v_begin, v_end, none_true),
                  "Return type of joint_none_of(Group g, Ptr first, Ptr last, "
                  "Predicate pred) is wrong\n");
              res_acc[8] =
                  sycl::joint_none_of(group, v_begin, v_end, none_true);
              res_acc[9] =
                  !sycl::joint_none_of(group, v_begin, v_end, one_true);
              res_acc[10] =
                  !sycl::joint_none_of(group, v_begin, v_end, some_true);
              res_acc[11] =
                  !sycl::joint_none_of(group, v_begin, v_end, all_true);

              sycl::sub_group sub_group = item.get_sub_group();

              ASSERT_RETURN_TYPE(
                  bool,
                  sycl::joint_any_of(sub_group, v_begin, v_end, none_true),
                  "Return type of joint_any_of(Sub_group g, Ptr first, Ptr "
                  "last, Predicate pred) is wrong\n");
              res_acc[12] =
                  !sycl::joint_any_of(sub_group, v_begin, v_end, none_true);
              res_acc[13] =
                  sycl::joint_any_of(sub_group, v_begin, v_end, one_true);
              res_acc[14] =
                  sycl::joint_any_of(sub_group, v_begin, v_end, some_true);
              res_acc[15] =
                  sycl::joint_any_of(sub_group, v_begin, v_end, all_true);

              ASSERT_RETURN_TYPE(
                  bool,
                  sycl::joint_all_of(sub_group, v_begin, v_end, none_true),
                  "Return type of joint_all_of(Sub_group g, Ptr first, Ptr "
                  "last, Predicate pred) is wrong\n");
              res_acc[16] =
                  !sycl::joint_all_of(sub_group, v_begin, v_end, none_true);
              res_acc[17] =
                  !sycl::joint_all_of(sub_group, v_begin, v_end, one_true);
              res_acc[18] =
                  !sycl::joint_all_of(sub_group, v_begin, v_end, some_true);
              res_acc[19] =
                  sycl::joint_all_of(sub_group, v_begin, v_end, all_true);

              ASSERT_RETURN_TYPE(
                  bool,
                  sycl::joint_none_of(sub_group, v_begin, v_end, none_true),
                  "Return type of joint_none_of(Sub_group g, Ptr first, Ptr "
                  "last, Predicate pred) is wrong\n");
              res_acc[20] =
                  sycl::joint_none_of(sub_group, v_begin, v_end, none_true);
              res_acc[21] =
                  !sycl::joint_none_of(sub_group, v_begin, v_end, one_true);
              res_acc[22] =
                  !sycl::joint_none_of(sub_group, v_begin, v_end, some_true);
              res_acc[23] =
                  !sycl::joint_none_of(sub_group, v_begin, v_end, all_true);
            });
      });
    }
    int index = 0;
    for (int i = 0; i < test_matrix; ++i)
      for (int j = 0; j < test_cases; ++j) {
        std::string work_group =
            sycl_cts::util::work_group_print(work_group_range);
        CAPTURE(D, work_group);
        INFO("Value of " << test_names[i] << " with " << test_cases_names[j]
                         << " predicate"
                            " is "
                         << (res[index] ? "right" : "wrong"));
        CHECK(res[index++]);
      }
  }
}

template <int D, typename T>
class predicate_function_of_group_kernel;

/**
 * @brief Provides test for group bool of operations with predicate functions
 * @tparam D Dimension to use for group instance
 * @tparam T Type pointed by Ptr
 */
template <int D, typename T>
void predicate_function_of_group(sycl::queue& queue) {
  // 3 functions * 4 predicates
  constexpr int test_matrix = 3;
  const std::string test_names[test_matrix] = {
      "bool any_of_group(group g, T x, Predicate pred)",
      "bool all_of_group(group g, T x, Predicate pred)",
      "bool none_of_group(group g, T x, Predicate pred)"};
  constexpr int test_cases = 4;
  const std::string test_cases_names[test_cases] = {"none true", "one true",
                                                    "some true", "all true"};

  sycl::range<D> work_group_range = sycl_cts::util::work_group_range<D>(queue);
  size_t work_group_size = work_group_range.size();
  // Note that the predicates 'one_true' and 'some_true' are in fact 'all_true'
  // for a work-group size of 1, but we don't consider that an interesting case
  // to test: assert to make this apparent to anything modifying the test case.
  assert(work_group_size != 1 &&
         "Not all test checks hold when the work-group size is 1");

  // array to return results: 4 predicates * 3 functions
  bool res[test_matrix * test_cases] = {false};
  {
    sycl::buffer<bool, 1> res_sycl(res,
                                   sycl::range<1>(test_matrix * test_cases));

    queue.submit([&](sycl::handler& cgh) {
      auto res_acc = res_sycl.get_access<sycl::access::mode::read_write>(cgh);

      sycl::nd_range<D> executionRange(work_group_range, work_group_range);

      cgh.parallel_for<predicate_function_of_group_kernel<D, T>>(
          executionRange, [=](sycl::nd_item<D> item) {
            sycl::group<D> group = item.get_group();

            T local_var(item.get_global_linear_id() + 1);

            // predicates
            auto none_true = [&](T i) { return i == 0; };
            auto one_true = [&](T i) { return i == 1; };
            auto some_true = [&](T i) { return i > work_group_size / 2; };
            auto all_true = [&](T i) { return i <= work_group_size; };

            ASSERT_RETURN_TYPE(bool,
                               sycl::any_of_group(group, local_var, none_true),
                               "Return type of any_of_group(group g, T x, "
                               "Predicate pred) is wrong\n");
            res_acc[0] = !sycl::any_of_group(group, local_var, none_true);
            res_acc[1] = sycl::any_of_group(group, local_var, one_true);
            res_acc[2] = sycl::any_of_group(group, local_var, some_true);
            res_acc[3] = sycl::any_of_group(group, local_var, all_true);

            ASSERT_RETURN_TYPE(bool,
                               sycl::all_of_group(group, local_var, none_true),
                               "Return type of any_of_group(group g, T x, "
                               "Predicate pred) is wrong\n");
            res_acc[4] = !sycl::all_of_group(group, local_var, none_true);
            res_acc[5] = !sycl::all_of_group(group, local_var, one_true);
            res_acc[6] = !sycl::all_of_group(group, local_var, some_true);
            res_acc[7] = sycl::all_of_group(group, local_var, all_true);

            ASSERT_RETURN_TYPE(bool,
                               sycl::none_of_group(group, local_var, none_true),
                               "Return type of none_of_group(group g, T x, "
                               "Predicate pred) is wrong\n");
            res_acc[8] = sycl::none_of_group(group, local_var, none_true);
            res_acc[9] = !sycl::none_of_group(group, local_var, one_true);
            res_acc[10] = !sycl::none_of_group(group, local_var, some_true);
            res_acc[11] = !sycl::none_of_group(group, local_var, all_true);
          });
    });
  }
  int index = 0;
  for (int i = 0; i < test_matrix; ++i)
    for (int j = 0; j < test_cases; ++j) {
      std::string work_group =
          sycl_cts::util::work_group_print(work_group_range);
      CAPTURE(D, work_group);
      INFO("Value of " << test_names[i] << " with " << test_cases_names[j]
                       << " predicate is " << (res[index] ? "right" : "wrong"));
      CHECK(res[index++]);
    }
}

template <int D, typename T>
class predicate_function_of_sub_group_kernel;

/**
 * @brief Provides test for sub_group bool of operations with predicate
 * functions
 * @tparam D Dimension to use for group instance
 * @tparam T Type pointed by Ptr
 */
template <int D, typename T>
void predicate_function_of_sub_group(sycl::queue& queue) {
  // 3 functions * 4 predicates
  constexpr int test_matrix = 3;
  const std::string test_names[test_matrix] = {
      "bool any_of_group(sub_group g, T x, Predicate pred)",
      "bool all_of_group(sub_group g, T x, Predicate pred)",
      "bool none_of_group(sub_group g, T x, Predicate pred)"};
  constexpr int test_cases = 4;
  const std::string test_cases_names[test_cases] = {"none true", "one true",
                                                    "some true", "all true"};

  sycl::range<D> work_group_range = sycl_cts::util::work_group_range<D>(queue);

  // array to return results: 4 predicates * 3 functions
  constexpr int total_case_count = test_matrix * test_cases;
  bool res[total_case_count];
  // Initially fill the results array with 'true'. Each sub-group test 'ands'
  // with this to ensure every sub-group in the work-group returns the correct
  // result.
  std::fill(res, res + total_case_count, true);
  {
    sycl::buffer<bool, 1> res_sycl(res, sycl::range<1>(total_case_count));

    queue.submit([&](sycl::handler& cgh) {
      auto res_acc = res_sycl.get_access<sycl::access::mode::read_write>(cgh);

      sycl::nd_range<D> executionRange(work_group_range, work_group_range);

      cgh.parallel_for<predicate_function_of_sub_group_kernel<
          D, T>>(executionRange, [=](sycl::nd_item<D> item) {
        sycl::sub_group sub_group = item.get_sub_group();
        T size = sub_group.get_local_linear_range();

        // Use the sub-group local ID (plus 1) as a variable against which to
        // test our predicates. Note that this has a well-defined set of values
        // [1,2,...,N] where N is the sub-group size. Note that the sub-group
        // could also just be of size 1.
        T local_var(sub_group.get_local_linear_id() + 1);

        // predicates
        // The variable is never 1 for any member of the sub-group
        auto none_true = [&](T i) { return i == 0; };
        // Exactly one member of the sub-group has value 1 (the first)
        auto one_true = [&](T i) { return i == 1; };
        // Some (or all, for sub-groups of size 1) members of the sub-group have
        // this value
        auto some_true = [&](T i) { return i > size / 2; };
        // The variable is less than or equal to the sub-group size for all
        // members of the sub-group.
        auto all_true = [&](T i) { return i <= size; };

        {
          ASSERT_RETURN_TYPE(
              bool, sycl::any_of_group(sub_group, local_var, none_true),
              "Return type of any_of_group(Sub_group g, bool pred) is wrong\n");
          res_acc[0] &= !sycl::any_of_group(sub_group, local_var, none_true);
          res_acc[1] &= sycl::any_of_group(sub_group, local_var, one_true);
          res_acc[2] &= sycl::any_of_group(sub_group, local_var, some_true);
          res_acc[3] &= sycl::any_of_group(sub_group, local_var, all_true);

          ASSERT_RETURN_TYPE(
              bool, sycl::all_of_group(sub_group, local_var, none_true),
              "Return type of all_of_group(Sub_group g, bool pred) is wrong\n");
          res_acc[4] &= !sycl::all_of_group(sub_group, local_var, none_true);
          // Note that 'one_true' returns true for the first item. Thus in the
          // case that the sub-group size is 1, check that all items match;
          // otherwise check that not all items match.
          res_acc[5] &=
              sycl::all_of_group(sub_group, local_var, one_true) ^ (size != 1);
          // Note that 'some_true' returns true for the first item if the
          // sub-group size is 1. In that case, check that all items match;
          // otherwise check that not all items match.
          res_acc[6] &=
              sycl::all_of_group(sub_group, local_var, some_true) ^ (size != 1);
          res_acc[7] &= sycl::all_of_group(sub_group, local_var, all_true);

          ASSERT_RETURN_TYPE(
              bool, sycl::none_of_group(sub_group, local_var, none_true),
              "Return type of none_of_group(Sub_group g, bool pred) is "
              "wrong\n");
          res_acc[8] &= sycl::none_of_group(sub_group, local_var, none_true);
          res_acc[9] &= !sycl::none_of_group(sub_group, local_var, one_true);
          res_acc[10] &= !sycl::none_of_group(sub_group, local_var, some_true);
          res_acc[11] &= !sycl::none_of_group(sub_group, local_var, all_true);
        }
      });
    });
  }
  int index = 0;
  for (int i = 0; i < test_matrix; ++i)
    for (int j = 0; j < test_cases; ++j) {
      std::string work_group =
          sycl_cts::util::work_group_print(work_group_range);
      CAPTURE(D, work_group);
      INFO("Value of " << test_names[i] << " with " << test_cases_names[j]
                       << " predicate is " << (res[index] ? "right" : "wrong"));
      CHECK(res[index++]);
    }
}

template <int D>
class predicate_function_of_group_bool_kernel;

/**
 * @brief Provides test for group bool of operations
 * @tparam D Dimension to use for group instance
 */
template <int D>
void bool_function_of_group(sycl::queue& queue) {
  // 3 functions * 4 predicates
  constexpr int test_matrix = 3;
  const std::string test_names[test_matrix] = {
      "bool any_of_group(group g, bool pred)",
      "bool all_of_group(group g, bool pred)",
      "bool none_of_group(group g, bool pred)"};
  constexpr int test_cases = 4;
  const std::string test_cases_names[test_cases] = {"none true", "one true",
                                                    "some true", "all true"};

  using T = size_t;

  sycl::range<D> work_group_range = sycl_cts::util::work_group_range<D>(queue);
  size_t work_group_size = work_group_range.size();
  // Note that the predicates 'one_true' and 'some_true' are in fact 'all_true'
  // for a work-group size of 1, but we don't consider that an interesting case
  // to test: assert to make this apparent to anyone modifying the test case.
  assert(work_group_size != 1 &&
         "Not all test checks hold when the work-group size is 1");

  // array to return results: 4 predicates * 3 functions
  bool res[test_matrix * test_cases] = {false};
  {
    sycl::buffer<bool, 1> res_sycl(res,
                                   sycl::range<1>(test_matrix * test_cases));

    queue.submit([&](sycl::handler& cgh) {
      auto res_acc = res_sycl.get_access<sycl::access::mode::read_write>(cgh);

      sycl::nd_range<D> executionRange(work_group_range, work_group_range);

      cgh.parallel_for<predicate_function_of_group_bool_kernel<D>>(
          executionRange, [=](sycl::nd_item<D> item) {
            sycl::group<D> group = item.get_group();
            T size = group.get_local_linear_range();

            T local_var(item.get_global_linear_id() + 1);

            // predicates
            auto none_true = [&](T i) { return i == 0; };
            auto one_true = [&](T i) { return i == 1; };
            auto some_true = [&](T i) { return i > size / 2; };
            auto all_true = [&](T i) { return i <= size; };

            ASSERT_RETURN_TYPE(
                bool, sycl::any_of_group(group, none_true(local_var)),
                "Return type of any_of_group(Group g, bool pred) is wrong\n");
            res_acc[0] = !sycl::any_of_group(group, none_true(local_var));
            res_acc[1] = sycl::any_of_group(group, one_true(local_var));
            res_acc[2] = sycl::any_of_group(group, some_true(local_var));
            res_acc[3] = sycl::any_of_group(group, all_true(local_var));

            ASSERT_RETURN_TYPE(
                bool, sycl::all_of_group(group, none_true(local_var)),
                "Return type of any_of_group(Group g, bool pred) is wrong\n");
            res_acc[4] = !sycl::all_of_group(group, none_true(local_var));
            res_acc[5] = !sycl::all_of_group(group, one_true(local_var));
            res_acc[6] = !sycl::all_of_group(group, some_true(local_var));
            res_acc[7] = sycl::all_of_group(group, all_true(local_var));

            ASSERT_RETURN_TYPE(
                bool, sycl::none_of_group(group, none_true(local_var)),
                "Return type of none_of_group(Group g, bool pred) is wrong\n");
            res_acc[8] = sycl::none_of_group(group, none_true(local_var));
            res_acc[9] = !sycl::none_of_group(group, one_true(local_var));
            res_acc[10] = !sycl::none_of_group(group, some_true(local_var));
            res_acc[11] = !sycl::none_of_group(group, all_true(local_var));
          });
    });
  }
  int index = 0;
  for (int i = 0; i < test_matrix; ++i)
    for (int j = 0; j < test_cases; ++j) {
      std::string work_group =
          sycl_cts::util::work_group_print(work_group_range);
      CAPTURE(D, work_group);
      INFO("Value of " << test_names[i] << " with " << test_cases_names[j]
                       << " predicate is " << (res[index] ? "right" : "wrong"));
      CHECK(res[index++]);
    }
}

template <int D>
class predicate_function_of_sub_group_bool_kernel;

/**
 * @brief Provides test for group bool of operations
 * @tparam D Dimension to use for group instance
 */
template <int D>
void bool_function_of_sub_group(sycl::queue& queue) {
  // 3 functions * 4 predicates
  constexpr int test_matrix = 3;
  const std::string test_names[test_matrix] = {
      "bool any_of_group(sub_group g, bool pred)",
      "bool all_of_group(sub_group g, bool pred)",
      "bool none_of_group(sub_group g, bool pred)"};
  constexpr int test_cases = 4;
  const std::string test_cases_names[test_cases] = {"none true", "one true",
                                                    "some true", "all true"};

  using T = size_t;

  sycl::range<D> work_group_range = sycl_cts::util::work_group_range<D>(queue);

  // array to return results: 4 predicates * 3 functions
  constexpr int total_case_count = test_matrix * test_cases;
  bool res[total_case_count];
  // Initially fill the results array with 'true'. Each sub-group test 'ands'
  // with this to ensure every sub-group in the work-group returns the correct
  // result.
  std::fill(res, res + total_case_count, true);
  {
    sycl::buffer<bool, 1> res_sycl(res, sycl::range<1>(total_case_count));

    queue.submit([&](sycl::handler& cgh) {
      auto res_acc = res_sycl.get_access<sycl::access::mode::read_write>(cgh);

      sycl::nd_range<D> executionRange(work_group_range, work_group_range);

      cgh.parallel_for<predicate_function_of_sub_group_bool_kernel<
          D>>(executionRange, [=](sycl::nd_item<D> item) {
        sycl::sub_group sub_group = item.get_sub_group();
        T size = sub_group.get_local_linear_range();

        // Use the sub-group local ID (plus 1) as a variable against which to
        // test our predicates. Note that this has a well-defined set of values
        // [1,2,...,N] where N is the sub-group size. Note that the sub-group
        // could also just be of size 1.
        T local_var(sub_group.get_local_linear_id() + 1);

        // predicates
        // The variable is never 0 for any member of the sub-group
        auto none_true = [&](T i) { return i == 0; };
        // Exactly one member of the sub-group has value 1 (the first)
        auto one_true = [&](T i) { return i == 1; };
        // Some (or all, for sub-groups of size 1) members of the sub-group have
        // this value
        auto some_true = [&](T i) { return i > size / 2; };
        // The variable is less than or equal to the sub-group size for all
        // members of the sub-group.
        auto all_true = [&](T i) { return i <= size; };

        {
          ASSERT_RETURN_TYPE(
              bool, sycl::any_of_group(sub_group, none_true(local_var)),
              "Return type of any_of_group(sub_group g, bool pred) is wrong\n");
          res_acc[0] &= !sycl::any_of_group(sub_group, none_true(local_var));
          res_acc[1] &= sycl::any_of_group(sub_group, one_true(local_var));
          res_acc[2] &= sycl::any_of_group(sub_group, some_true(local_var));
          res_acc[3] &= sycl::any_of_group(sub_group, all_true(local_var));

          ASSERT_RETURN_TYPE(
              bool, sycl::all_of_group(sub_group, none_true(local_var)),
              "Return type of all_of_group(sub_group g, bool pred) is wrong\n");
          res_acc[4] = !sycl::all_of_group(sub_group, none_true(local_var));
          // Note that 'one_true' returns true for the first item. Thus in the
          // case that the sub-group size is 1, check that all items match;
          // otherwise check that not all items match.
          res_acc[5] &=
              sycl::all_of_group(sub_group, one_true(local_var)) ^ (size != 1);
          // Note that 'some_true' returns true for the first item if the
          // sub-group size is 1. In that case, check that all items match;
          // otherwise check that not all items match.
          res_acc[6] &=
              sycl::all_of_group(sub_group, some_true(local_var)) ^ (size != 1);
          res_acc[7] &= sycl::all_of_group(sub_group, all_true(local_var));

          ASSERT_RETURN_TYPE(
              bool, sycl::none_of_group(sub_group, none_true(local_var)),
              "Return type of none_of_group(sub_group g, bool pred) is "
              "wrong\n");
          res_acc[8] &= sycl::none_of_group(sub_group, none_true(local_var));
          res_acc[9] &= !sycl::none_of_group(sub_group, one_true(local_var));
          res_acc[10] &= !sycl::none_of_group(sub_group, some_true(local_var));
          res_acc[11] &= !sycl::none_of_group(sub_group, all_true(local_var));
        }
      });
    });
  }
  int index = 0;
  for (int i = 0; i < test_matrix; ++i)
    for (int j = 0; j < test_cases; ++j) {
      std::string work_group =
          sycl_cts::util::work_group_print(work_group_range);
      CAPTURE(D, work_group);
      INFO("Value of " << test_names[i] << " with " << test_cases_names[j]
                       << " predicate is " << (res[index] ? "right" : "wrong"));
      CHECK(res[index++]);
    }
}

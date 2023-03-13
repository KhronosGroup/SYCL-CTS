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

#include "../common/disabled_for_test_case.h"

#include "group_functions_common.h"

template <int D, typename T>
class joint_reduce_group_kernel;

/**
 * @brief Provides test for joint reduce by group
 * @tparam D Dimension to use for group instance
 * @tparam T Type for reduced values
 */
template <int D, typename T>
void joint_reduce_group(sycl::queue& queue) {
  // 2 functions * 2 function objects
  constexpr int test_matrix = 2;
  const std::string test_names[test_matrix] = {
      "std::iterator_traits<Ptr>::value_type joint_reduce(group g, Ptr first, "
      "Ptr last, BinaryOperation binary_op)",
      "std::iterator_traits<Ptr>::value_type joint_reduce(sub_group g, Ptr "
      "first, Ptr last, BinaryOperation binary_op)"};
  constexpr int test_cases = 2;
  const std::string test_cases_names[test_cases] = {"plus", "maximum"};

  sycl::range<D> work_group_range = sycl_cts::util::work_group_range<D>(queue);
  size_t work_group_size = work_group_range.size();

  const size_t sizes[3] = {5, work_group_size / 2, 3 * work_group_size};
  for (size_t size : sizes) {
    std::vector<T> v(size);
    std::iota(v.begin(), v.end(), 1);

    // array to return results
    bool res[test_matrix * test_cases] = {false};
    {
      sycl::buffer<T, 1> v_sycl(v.data(), sycl::range<1>(size));
      sycl::buffer<bool, 1> res_sycl(res,
                                     sycl::range<1>(test_matrix * test_cases));

      queue.submit([&](sycl::handler& cgh) {
        auto v_acc = v_sycl.template get_access<sycl::access::mode::read>(cgh);
        auto res_acc = res_sycl.get_access<sycl::access::mode::read_write>(cgh);

        sycl::nd_range<D> executionRange(work_group_range, work_group_range);

        cgh.parallel_for<joint_reduce_group_kernel<D, T>>(
            executionRange, [=](sycl::nd_item<D> item) {
              T* v_begin = v_acc.get_pointer();
              T* v_end = v_begin + v_acc.size();
              // checks are plagued by UB for too short types
              // so that the guards are introduced as the second parts in
              // res_acc calculations
              size_t reduced;

              sycl::group<D> group = item.get_group();

              ASSERT_RETURN_TYPE(
                  T, sycl::joint_reduce(group, v_begin, v_end, sycl::plus<T>()),
                  "Return type of joint_reduce(group g, Ptr first, Ptr last, "
                  "BinaryOperation binary_op) is wrong\n");

              reduced = size * (size + 1) / 2;
              res_acc[0] = (reduced == sycl::joint_reduce(group, v_begin, v_end,
                                                          sycl::plus<T>())) ||
                           (reduced > util::exact_max<T>);

              reduced = size;
              res_acc[1] =
                  (reduced == sycl::joint_reduce(group, v_begin, v_end,
                                                 sycl::maximum<T>())) ||
                  (reduced > util::exact_max<T>);

              sycl::sub_group sub_group = item.get_sub_group();

              ASSERT_RETURN_TYPE(
                  T,
                  sycl::joint_reduce(sub_group, v_begin, v_end,
                                     sycl::maximum<T>()),
                  "Return type of joint_reduce(sub_group g, Ptr first, Ptr "
                  "last, BinaryOperation binary_op) is wrong\n");

              reduced = size * (size + 1) / 2;
          // FIXME: hipSYCL has no implementation over sub-groups
#ifdef SYCL_CTS_COMPILING_WITH_HIPSYCL
              res_acc[2] = true;
#else
            res_acc[2] = (reduced == sycl::joint_reduce(sub_group, v_begin, v_end, sycl::plus<T>()))
              || (reduced > util::exact_max<T>);
#endif

              reduced = size;
          // FIXME: hipSYCL has no implementation over sub-groups
#ifdef SYCL_CTS_COMPILING_WITH_HIPSYCL
              res_acc[3] = true;
#else
            res_acc[3] = (reduced == sycl::joint_reduce(sub_group, v_begin, v_end, sycl::maximum<T>()))
              || (reduced > util::exact_max<T>);
#endif
            });
      });
    }
    int index = 0;
    for (int i = 0; i < test_matrix; ++i)
      for (int j = 0; j < test_cases; ++j) {
        std::string work_group =
            sycl_cts::util::work_group_print(work_group_range);
        CAPTURE(D, work_group, size);
        INFO("Value of " << test_names[i] << " with " << test_cases_names[j]
                         << " operation"
                            " and Ptr = "
                         << type_name<T>() << "* is "
                         << (res[index] ? "right" : "wrong"));
        CHECK(res[index++]);
      }
  }
}

template <int D, typename T, typename U>
class init_joint_reduce_group_kernel;

/**
 * @brief Provides test for joint reduce by group with init
 * @tparam D Dimension to use for group instance
 * @tparam T Type for init and result values
 * @tparam U Type for reduced values
 */
template <int D, typename T, typename U>
void init_joint_reduce_group(sycl::queue& queue) {
  // 2 functions * 2 function objects
  constexpr int test_matrix = 2;
  const std::string test_names[test_matrix] = {
      "T joint_reduce(group g, Ptr first, Ptr last, T init, BinaryOperation "
      "binary_op)",
      "T joint_reduce(sub_group g, Ptr first, Ptr last, T init, "
      "BinaryOperation binary_op)"};
  constexpr int test_cases = 2;
  const std::string test_cases_names[test_cases] = {"plus", "maximum"};

  sycl::range<D> work_group_range = sycl_cts::util::work_group_range<D>(queue);
  size_t work_group_size = work_group_range.size();

  const size_t sizes[3] = {5, work_group_size / 2, 3 * work_group_size};
  for (size_t size : sizes) {
    std::vector<U> v(size);
    std::iota(v.begin(), v.end(), 1);

    // array to return results
    bool res[test_matrix * test_cases] = {false};
    {
      sycl::buffer<U, 1> v_sycl(v.data(), sycl::range<1>(size));
      sycl::buffer<bool, 1> res_sycl(res,
                                     sycl::range<1>(test_matrix * test_cases));

      queue.submit([&](sycl::handler& cgh) {
        auto v_acc = v_sycl.template get_access<sycl::access::mode::read>(cgh);
        auto res_acc = res_sycl.get_access<sycl::access::mode::read_write>(cgh);

        sycl::nd_range<D> executionRange(work_group_range, work_group_range);

        cgh.parallel_for<init_joint_reduce_group_kernel<D, T, U>>(
            executionRange, [=](sycl::nd_item<D> item) {
              sycl::group<D> group = item.get_group();
              sycl::sub_group sub_group = item.get_sub_group();

              U* v_begin = v_acc.get_pointer();
              U* v_end = v_begin + v_acc.size();
              // checks are plagued by UB for too short types
              // so that the guards are introduced as the second parts in
              // res_acc calculations
              size_t reduced;

              ASSERT_RETURN_TYPE(
                  T,
                  sycl::joint_reduce(group, v_begin, v_end, T(1412),
                                     sycl::plus<T>()),
                  "Return type of joint_reduce(group g, Ptr first, Ptr last, T "
                  "init, BinaryOperation binary_op) is wrong\n");

              reduced = size * (size + 1) / 2 + T(1412);
              res_acc[0] =
                  (reduced == sycl::joint_reduce(group, v_begin, v_end, T(1412),
                                                 sycl::plus<T>())) ||
                  (reduced > util::exact_max<T>) || (size > util::exact_max<U>);

              reduced = 2 * size;
              res_acc[1] =
                  (reduced == sycl::joint_reduce(group, v_begin, v_end,
                                                 T(2 * size),
                                                 sycl::maximum<T>())) ||
                  (reduced > util::exact_max<T>) || (size > util::exact_max<U>);

              ASSERT_RETURN_TYPE(
                  T,
                  sycl::joint_reduce(sub_group, v_begin, v_end, T(1412),
                                     sycl::maximum<T>()),
                  "Return type of joint_reduce(sub_group g, Ptr first, Ptr "
                  "last, T init, BinaryOperation binary_op) is wrong\n");

              reduced = size * (size + 1) / 2 + T(1412);
          // FIXME: hipSYCL has no implementation over sub-groups
#ifdef SYCL_CTS_COMPILING_WITH_HIPSYCL
              res_acc[2] = true;
#else
            res_acc[2] = (reduced == sycl::joint_reduce(sub_group, v_begin, v_end, T(1412), sycl::plus<T>()))
              || (reduced > util::exact_max<T>) || (size > util::exact_max<U>);
#endif

              reduced = 2 * size;
          // FIXME: hipSYCL has no implementation over sub-groups
#ifdef SYCL_CTS_COMPILING_WITH_HIPSYCL
              res_acc[3] = true;
#else
            res_acc[3] = (reduced == sycl::joint_reduce(sub_group, v_begin, v_end, T(2 * size), sycl::maximum<T>()))
              || (reduced > util::exact_max<T>) || (size > util::exact_max<U>);
#endif
            });
      });
    }
    int index = 0;
    for (int i = 0; i < test_matrix; ++i)
      for (int j = 0; j < test_cases; ++j) {
        std::string work_group =
            sycl_cts::util::work_group_print(work_group_range);
        CAPTURE(D, work_group, size);
        INFO("Value of " << test_names[i] << " with " << test_cases_names[j]
                         << " operation"
                            " and Ptr = "
                         << type_name<U>() << "*, T = " << type_name<T>()
                         << " is " << (res[index] ? "right" : "wrong"));
        CHECK(res[index++]);
      }
  }
}

template <int D, typename T>
class reduce_over_group_kernel;

/**
 * @brief Provides test for reduce over group values
 * @tparam D Dimension to use for group instance
 * @tparam T Type for reduced values
 */
template <int D, typename T>
void reduce_over_group(sycl::queue& queue) {
  // 2 function * 2 function objects
  constexpr int test_matrix = 2;
  const std::string test_names[test_matrix] = {
      "T reduce_over_group(group g, T x, BinaryOperation binary_op)",
      "T reduce_over_group(sub_group g, T x, BinaryOperation binary_op)"};
  constexpr int test_cases = 2;
  const std::string test_cases_names[test_cases] = {"plus", "maximum"};

  sycl::range<D> work_group_range = sycl_cts::util::work_group_range<D>(queue);

  // array to return results
  bool res[test_matrix * test_cases] = {false};
  {
    sycl::buffer<bool, 1> res_sycl(res,
                                   sycl::range<1>(test_matrix * test_cases));

    queue.submit([&](sycl::handler& cgh) {
      auto res_acc = res_sycl.get_access<sycl::access::mode::read_write>(cgh);

      sycl::nd_range<D> executionRange(work_group_range, work_group_range);

      cgh.parallel_for<reduce_over_group_kernel<D, T>>(
          executionRange, [=](sycl::nd_item<D> item) {
            sycl::group<D> group = item.get_group();
            size_t group_size = group.get_local_linear_range();

            T local_var = item.get_local_linear_id() + 1;
            // checks are plagued by UB for too short types
            // so that the guards are introduced as the second parts in res_acc
            // calculations
            size_t reduced;

            ASSERT_RETURN_TYPE(
                T, sycl::reduce_over_group(group, local_var, sycl::plus<T>()),
                "Return type of reduce_over_group(group g, T x, "
                "BinaryOperation binary_op) is wrong\n");

            reduced = group_size * (group_size + 1) / 2;
            res_acc[0] = (reduced == sycl::reduce_over_group(
                                         group, local_var, sycl::plus<T>())) ||
                         (reduced > util::exact_max<T>);

            reduced = group_size;
            res_acc[1] =
                (reduced == sycl::reduce_over_group(group, local_var,
                                                    sycl::maximum<T>())) ||
                (reduced > util::exact_max<T>);

            sycl::sub_group sub_group = item.get_sub_group();
            size_t sub_group_size = sub_group.get_local_linear_range();

            local_var = sub_group.get_local_linear_id() + 1;

            ASSERT_RETURN_TYPE(T,
                               sycl::reduce_over_group(sub_group, local_var,
                                                       sycl::maximum<T>()),
                               "Return type of reduce_over_group(sub_group g, "
                               "T x, BinaryOperation binary_op) is wrong\n");

            reduced = sub_group_size * (sub_group_size + 1) / 2;
            res_acc[2] =
                (reduced == sycl::reduce_over_group(sub_group, local_var,
                                                    sycl::plus<T>())) ||
                (reduced > util::exact_max<T>);

            reduced = sub_group_size;
            res_acc[3] =
                (reduced == sycl::reduce_over_group(sub_group, local_var,
                                                    sycl::maximum<T>())) ||
                (reduced > util::exact_max<T>);
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
                       << " operation"
                          " and T = "
                       << type_name<T>() << " is "
                       << (res[index] ? "right" : "wrong"));
      CHECK(res[index++]);
    }
}

template <int D, typename T, typename U>
class init_reduce_over_group_kernel;

/**
 * @brief Provides test for reduce over group values with init
 * @tparam D Dimension to use for group instance
 * @tparam T Type for group values
 * @tparam U Type for init and result values
 */
template <int D, typename T, typename U>
void init_reduce_over_group(sycl::queue& queue) {
  // 2 function * 2 function objects
  constexpr int test_matrix = 2;
  const std::string test_names[test_matrix] = {
      "T reduce_over_group(group g, V x, T init, BinaryOperation binary_op)",
      "T reduce_over_group(sub_group g, V x, T init, BinaryOperation "
      "binary_op)"};
  constexpr int test_cases = 2;
  const std::string test_cases_names[test_cases] = {"plus", "maximum"};

  sycl::range<D> work_group_range = sycl_cts::util::work_group_range<D>(queue);

  // array to return results
  bool res[test_matrix * test_cases] = {false};
  {
    sycl::buffer<bool, 1> res_sycl(res,
                                   sycl::range<1>(test_matrix * test_cases));

    queue.submit([&](sycl::handler& cgh) {
      auto res_acc = res_sycl.get_access<sycl::access::mode::read_write>(cgh);

      sycl::nd_range<D> executionRange(work_group_range, work_group_range);

      cgh.parallel_for<init_reduce_over_group_kernel<D, T, U>>(
          executionRange, [=](sycl::nd_item<D> item) {
            sycl::group<D> group = item.get_group();
            size_t group_size = group.get_local_linear_range();

            U local_var = item.get_local_linear_id() + 1;
            // checks are plagued by UB for too short types
            // so that the guards are introduced as the second parts in res_acc
            // calculations
            size_t reduced;

            ASSERT_RETURN_TYPE(T,
                               sycl::reduce_over_group(
                                   group, local_var, T(1412), sycl::plus<T>()),
                               "Return type of reduce_over_group(group g, V x, "
                               "T init, BinaryOperation binary_op) is wrong\n");

            reduced = group_size * (group_size + 1) / 2 + T(1412);
            res_acc[0] =
                (reduced == sycl::reduce_over_group(group, local_var, T(1412),
                                                    sycl::plus<T>())) ||
                (group_size > util::exact_max<U>) ||
                (reduced > util::exact_max<T>);

            reduced = 2 * group_size;
            res_acc[1] = (reduced == sycl::reduce_over_group(
                                         group, local_var, T(2 * group_size),
                                         sycl::maximum<T>())) ||
                         (group_size > util::exact_max<U>) ||
                         (reduced > util::exact_max<T>);

            sycl::sub_group sub_group = item.get_sub_group();
            size_t sub_group_size = sub_group.get_local_linear_range();

            local_var = sub_group.get_local_linear_id() + 1;

            ASSERT_RETURN_TYPE(
                T,
                sycl::reduce_over_group(sub_group, local_var, T(1412),
                                        sycl::maximum<T>()),
                "Return type of reduce_over_group(sub_group g, V x, T init, "
                "BinaryOperation binary_op) is wrong\n");

            reduced = sub_group_size * (sub_group_size + 1) / 2 + T(1412);
            res_acc[2] = (reduced ==
                          sycl::reduce_over_group(sub_group, local_var, T(1412),
                                                  sycl::plus<T>())) ||
                         (sub_group_size > util::exact_max<U>) ||
                         (reduced > util::exact_max<T>);

            reduced = 2 * sub_group_size;
            res_acc[3] =
                (reduced == sycl::reduce_over_group(sub_group, local_var,
                                                    T(2 * sub_group_size),
                                                    sycl::maximum<T>())) ||
                (sub_group_size > util::exact_max<U>) ||
                (reduced > util::exact_max<T>);
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
                       << " operation"
                          " and T = "
                       << type_name<U>() << ", V = " << type_name<T>() << " is "
                       << (res[index] ? "right" : "wrong"));
      CHECK(res[index++]);
    }
}

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

template <int D, typename T, typename U>
class joint_scan_group_kernel;

/**
 * @brief Provides test for joint scans
 * @tparam D Dimension to use for group instance
 * @tparam T Type pointed by InPtr
 * @tparam U Type pointed by OutPtr
 */
template <int D, typename T, typename U>
void joint_scan_group(sycl::queue& queue) {
  // 4 functions * 2 function objects
  constexpr int test_matrix = 4;
  const std::string test_names[test_matrix] = {
      "OutPtr joint_exclusive_scan(group g, InPtr first, InPtr last, OutPtr "
      "result, BinaryOperation binary_op)",
      "OutPtr joint_inclusive_scan(group g, InPtr first, InPtr last, OutPtr "
      "result, BinaryOperation binary_op)",
      "OutPtr joint_exclusive_scan(sub_group g, InPtr first, InPtr last, "
      "OutPtr result, BinaryOperation binary_op)",
      "OutPtr joint_inclusive_scan(sub_group g, InPtr first, InPtr last, "
      "OutPtr result, BinaryOperation binary_op)"};
  constexpr int test_cases = 2;
  const std::string test_cases_names[test_cases] = {"plus", "maximum"};

  sycl::range<D> work_group_range = util::work_group_range<D>(queue);
  size_t work_group_size = work_group_range.size();

  const size_t sizes[3] = {5, work_group_size / 2, 3 * work_group_size};
  for (size_t size : sizes) {
    std::vector<T> v(size);
    std::iota(v.begin(), v.end(), T(1));
    std::vector<U> r(size, U(-1));

    // array to return results
    bool res[test_matrix * test_cases] = {false};
    {
      sycl::buffer<T, 1> v_sycl(v.data(), sycl::range<1>(size));
      sycl::buffer<U, 1> r_sycl(r.data(), sycl::range<1>(size));

      sycl::buffer<bool, 1> res_sycl(res,
                                     sycl::range<1>(test_matrix * test_cases));

      queue.submit([&](sycl::handler& cgh) {
        auto v_acc = v_sycl.template get_access<sycl::access::mode::read>(cgh);
        auto r_acc =
            r_sycl.template get_access<sycl::access::mode::read_write>(cgh);
        auto res_acc = res_sycl.get_access<sycl::access::mode::read_write>(cgh);

        sycl::nd_range<D> executionRange(work_group_range, work_group_range);

        cgh.parallel_for<joint_scan_group_kernel<
            D, T, U>>(executionRange, [=](sycl::nd_item<D> item) {
          T* v_begin = v_acc.get_pointer();
          T* v_end = v_begin + v_acc.size();
          U* r_begin = r_acc.get_pointer();
          ;
          U* r_end;

          auto op_plus = sycl::plus<U>();
          auto op_max = sycl::maximum<U>();

          sycl::group<D> group = item.get_group();

          ASSERT_RETURN_TYPE(
              U*,
              sycl::joint_exclusive_scan(group, v_begin, v_end, r_begin,
                                         op_plus),
              "Return type of joint_exclusive_scan(group g, InPtr first, InPtr "
              "last, OutPtr result, BinaryOperation binary_op) is wrong\n");

          r_end = sycl::joint_exclusive_scan(group, v_begin, v_end, r_begin,
                                             op_plus);
          if (group.get_local_linear_id() == 0) {
            bool check = (r_begin + r_acc.size() == r_end);
            // FIXME: hipSYCL does not support sycl::known_identity_v yet
#if SYCL_CTS_COMPILING_WITH_HIPSYCL
            U scan = U{};
#else
              U scan = sycl::known_identity_v<sycl::plus<U>, U>;
#endif
            for (int i = 0; i < size; ++i) {
              check &= (scan == r_begin[i]);
              scan = op_plus(scan, v_begin[i]);
            }
            res_acc[0] = check;
          }

          r_end = sycl::joint_exclusive_scan(group, v_begin, v_end, r_begin,
                                             op_max);
          if (group.get_local_linear_id() == 0) {
            bool check = (r_begin + r_acc.size() == r_end);
            // FIXME: hipSYCL does not support sycl::known_identity_v yet
#if SYCL_CTS_COMPILING_WITH_HIPSYCL
            U scan = std::numeric_limits<U>::lowest();
#else
              U scan = sycl::known_identity_v<sycl::maximum<U>, U>;
#endif
            for (int i = 0; i < size; ++i) {
              check &= (scan == r_begin[i]);
              scan = op_max(scan, v_begin[i]);
            }
            res_acc[1] = check;
          }

          ASSERT_RETURN_TYPE(
              U*,
              sycl::joint_inclusive_scan(group, v_begin, v_end, r_begin,
                                         op_plus),
              "Return type of joint_inclusive_scan(group g, InPtr first, InPtr "
              "last, OutPtr result, BinaryOperation binary_op) is wrong\n");

          r_end = sycl::joint_inclusive_scan(group, v_begin, v_end, r_begin,
                                             op_plus);
          if (group.get_local_linear_id() == 0) {
            bool check = (r_begin + r_acc.size() == r_end);
            U scan = U{};
            for (int i = 0; i < size; ++i) {
              scan = op_plus(scan, v_begin[i]);
              check &= (scan == r_begin[i]);
            }
            res_acc[2] = check;
          }

          r_end = sycl::joint_inclusive_scan(group, v_begin, v_end, r_begin,
                                             op_max);
          if (group.get_local_linear_id() == 0) {
            bool check = (r_begin + r_acc.size() == r_end);
            U scan = v_begin[0];
            for (int i = 0; i < size; ++i) {
              scan = op_max(scan, v_begin[i]);
              check &= (scan == r_begin[i]);
            }
            res_acc[3] = check;
          }

          sycl::sub_group sub_group = item.get_sub_group();

          // FIXME: it should work without (all sub-groups do the same), but in
          // hipSYCL it leads to errors in the test above (sic!) for res_acc[3]
#if SYCL_CTS_COMPILING_WITH_HIPSYCL
          if (sub_group.get_group_linear_id() == 0)
#endif
          {
            ASSERT_RETURN_TYPE(U*,
                               sycl::joint_exclusive_scan(
                                   sub_group, v_begin, v_end, r_begin, op_plus),
                               "Return type of joint_exclusive_scan(sub_group "
                               "g, InPtr first, InPtr last, OutPtr result, "
                               "BinaryOperation binary_op) is wrong\n");

            r_end = sycl::joint_exclusive_scan(sub_group, v_begin, v_end,
                                               r_begin, op_plus);
            if (sub_group.get_local_linear_id() == 0) {
              bool check = (r_begin + r_acc.size() == r_end);
              // FIXME: hipSYCL does not support sycl::known_identity_v yet
#if SYCL_CTS_COMPILING_WITH_HIPSYCL
              U scan = U{};
#else
              U scan = sycl::known_identity_v<sycl::plus<U>, U>;
#endif
              for (int i = 0; i < size; ++i) {
                check &= (scan == r_begin[i]);
                scan = op_plus(scan, v_begin[i]);
              }
              res_acc[4] = check;
            }

            r_end = sycl::joint_exclusive_scan(sub_group, v_begin, v_end,
                                               r_begin, op_max);
            if (sub_group.get_local_linear_id() == 0) {
              bool check = (r_begin + r_acc.size() == r_end);
              // FIXME: hipSYCL does not support sycl::known_identity_v yet
#if SYCL_CTS_COMPILING_WITH_HIPSYCL
              U scan = std::numeric_limits<U>::lowest();
#else
              U scan = sycl::known_identity_v<sycl::maximum<U>, U>;
#endif
              for (int i = 0; i < size; ++i) {
                check &= (scan == r_begin[i]);
                scan = op_max(scan, v_begin[i]);
              }
              res_acc[5] = check;
            }

            ASSERT_RETURN_TYPE(U*,
                               sycl::joint_inclusive_scan(
                                   sub_group, v_begin, v_end, r_begin, op_plus),
                               "Return type of joint_inclusive_scan(sub_group "
                               "g, InPtr first, InPtr last, OutPtr result, "
                               "BinaryOperation binary_op) is wrong\n");

            r_end = sycl::joint_inclusive_scan(sub_group, v_begin, v_end,
                                               r_begin, op_plus);
            if (sub_group.get_local_linear_id() == 0) {
              bool check = (r_begin + r_acc.size() == r_end);
              U scan = U{};
              for (int i = 0; i < size; ++i) {
                scan = op_plus(scan, v_begin[i]);
                check &= (scan == r_begin[i]);
              }
              res_acc[6] = check;
            }

            r_end = sycl::joint_inclusive_scan(sub_group, v_begin, v_end,
                                               r_begin, op_max);
            if (sub_group.get_local_linear_id() == 0) {
              bool check = (r_begin + r_acc.size() == r_end);
              U scan = v_begin[0];
              for (int i = 0; i < size; ++i) {
                scan = op_max(scan, v_begin[i]);
                check &= (scan == r_begin[i]);
              }
              res_acc[7] = check;
            }
          }
        });
      });
    }
    int index = 0;
    for (int i = 0; i < test_matrix; ++i)
      for (int j = 0; j < test_cases; ++j) {
        std::string work_group = util::work_group_print(work_group_range);
        CAPTURE(D, work_group, size);
        INFO("Value of " << test_names[i] << " with " << test_cases_names[j]
                         << " operation"
                            " and In = "
                         << type_name<T>() << ", Out = " << type_name<U>()
                         << " is " << (res[index] ? "right" : "wrong"));
        CHECK(res[index++]);
      }
  }
}

template <typename DimensionT, typename T, typename U>
class invoke_joint_scan_group {
  static constexpr int D = DimensionT::value;

 public:
  void operator()(sycl::queue& queue) { joint_scan_group<D, T, U>(queue); }
};

// FIXME: Helper for implementations that cannot handle cases of different types
template <typename DimensionT, typename T>
class invoke_joint_scan_group_same_type {
  static constexpr int D = DimensionT::value;

 public:
  void operator()(sycl::queue& queue) { joint_scan_group<D, T, T>(queue); }
};

template <int D, typename T, typename U, typename I>
class init_joint_scan_group_kernel;

/**
 * @brief Provides test for joint scans with init
 * @tparam D Dimension to use for group instance
 * @tparam T Type pointed by InPtr
 * @tparam U Type pointed by OutPtr
 * @tparam I Type used for init value
 */
template <int D, typename T, typename U, typename I>
void init_joint_scan_group(sycl::queue& queue) {
  // 4 functions * 2 function objects
  constexpr int test_matrix = 4;
  const std::string test_names[test_matrix] = {
      "OutPtr joint_exclusive_scan(group g, InPtr first, InPtr last, OutPtr "
      "result, T init, BinaryOperation binary_op)",
      "OutPtr joint_inclusive_scan(group g, InPtr first, InPtr last, OutPtr "
      "result, BinaryOperation binary_op, T init)",
      "OutPtr joint_exclusive_scan(sub_group g, InPtr first, InPtr last, "
      "OutPtr result, T init, BinaryOperation binary_op)",
      "OutPtr joint_inclusive_scan(sub_group g, InPtr first, InPtr last, "
      "OutPtr result, BinaryOperation binary_op, T init)"};
  constexpr int test_cases = 2;
  const std::string test_cases_names[test_cases] = {"plus", "maximum"};

  sycl::range<D> work_group_range = util::work_group_range<D>(queue);
  size_t work_group_size = work_group_range.size();

  const size_t sizes[3] = {5, work_group_size / 2, 3 * work_group_size};
  for (size_t size : sizes) {
    std::vector<T> v(size);
    std::iota(v.begin(), v.end(), 1);
    std::vector<U> r(size, U(-1));

    // array to return results
    bool res[test_matrix * test_cases] = {false};
    {
      sycl::buffer<T, 1> v_sycl(v.data(), sycl::range<1>(size));
      sycl::buffer<U, 1> r_sycl(r.data(), sycl::range<1>(size));

      sycl::buffer<bool, 1> res_sycl(res,
                                     sycl::range<1>(test_matrix * test_cases));

      queue.submit([&](sycl::handler& cgh) {
        auto v_acc = v_sycl.template get_access<sycl::access::mode::read>(cgh);
        auto r_acc =
            r_sycl.template get_access<sycl::access::mode::read_write>(cgh);
        auto res_acc = res_sycl.get_access<sycl::access::mode::read_write>(cgh);

        sycl::nd_range<D> executionRange(work_group_range, work_group_range);

        cgh.parallel_for<init_joint_scan_group_kernel<D, T, U, I>>(
            executionRange, [=](sycl::nd_item<D> item) {
              T* v_begin = v_acc.get_pointer();
              T* v_end = v_begin + v_acc.size();
              U* r_begin = r_acc.get_pointer();
              ;
              U* r_end;

              auto op_plus = sycl::plus<I>();
              auto op_max = sycl::maximum<I>();

              sycl::group<D> group = item.get_group();

              ASSERT_RETURN_TYPE(
                  U*,
                  sycl::joint_exclusive_scan(group, v_begin, v_end, r_begin,
                                             I(1412), op_max),
                  "Return type of joint_exclusive_scan(group g, InPtr first, "
                  "InPtr last, OutPtr result, T init, BinaryOperation "
                  "binary_op) is wrong\n");

              r_end = sycl::joint_exclusive_scan(group, v_begin, v_end, r_begin,
                                                 I(1412), op_plus);
              if (group.get_local_linear_id() == 0) {
                bool check = (r_begin + r_acc.size() == r_end);
                I scan = I(1412);
                for (int i = 0; i < size; ++i) {
                  check &= (U(scan) == r_begin[i]);
                  scan = op_plus(scan, v_begin[i]);
                }
                res_acc[0] = check;
              }

              r_end = sycl::joint_exclusive_scan(group, v_begin, v_end, r_begin,
                                                 I(42), op_max);
              if (group.get_local_linear_id() == 0) {
                bool check = (r_begin + r_acc.size() == r_end);
                I scan = I(42);
                for (int i = 0; i < size; ++i) {
                  check &= (U(scan) == r_begin[i]);
                  scan = op_max(scan, v_begin[i]);
                }
                res_acc[1] = check;
              }

              ASSERT_RETURN_TYPE(
                  U*,
                  sycl::joint_inclusive_scan(group, v_begin, v_end, r_begin,
                                             op_plus, I(1412)),
                  "Return type of joint_inclusive_scan(group g, InPtr first, "
                  "InPtr last, OutPtr result, BinaryOperation binary_op, T "
                  "init) is wrong\n");

              r_end = sycl::joint_inclusive_scan(group, v_begin, v_end, r_begin,
                                                 op_plus, I(1412));
              if (group.get_local_linear_id() == 0) {
                bool check = (r_begin + r_acc.size() == r_end);
                I scan = I(1412);
                for (int i = 0; i < size; ++i) {
                  scan = op_plus(scan, v_begin[i]);
                  check &= (U(scan) == r_begin[i]);
                }
                res_acc[2] = check;
              }

              r_end = sycl::joint_inclusive_scan(group, v_begin, v_end, r_begin,
                                                 op_max, I(42));
              if (group.get_local_linear_id() == 0) {
                bool check = (r_begin + r_acc.size() == r_end);
                I scan = I(42);
                for (int i = 0; i < size; ++i) {
                  scan = op_max(scan, v_begin[i]);
                  check &= (U(scan) == r_begin[i]);
                }
                res_acc[3] = check;
              }

              sycl::sub_group sub_group = item.get_sub_group();

          // FIXME: it should work without (all sub-groups do the same), but in
          // hipSYCL it leads to errors in the test above (sic!) for res_acc[3]
#if SYCL_CTS_COMPILING_WITH_HIPSYCL
              if (sub_group.get_group_linear_id() == 0)
#endif
              {
                ASSERT_RETURN_TYPE(
                    U*,
                    sycl::joint_exclusive_scan(sub_group, v_begin, v_end,
                                               r_begin, I(1412), op_max),
                    "Return type of joint_exclusive_scan(sub_group g, InPtr "
                    "first, InPtr last, OutPtr result, T init, BinaryOperation "
                    "binary_op) is wrong\n");

                r_end = sycl::joint_exclusive_scan(sub_group, v_begin, v_end,
                                                   r_begin, I(1412), op_plus);
                if (sub_group.get_local_linear_id() == 0) {
                  bool check = (r_begin + r_acc.size() == r_end);
                  I scan = I(1412);
                  for (int i = 0; i < size; ++i) {
                    check &= (U(scan) == r_begin[i]);
                    scan = op_plus(scan, v_begin[i]);
                  }
                  res_acc[4] = check;
                }

                r_end = sycl::joint_exclusive_scan(sub_group, v_begin, v_end,
                                                   r_begin, I(4), op_max);
                if (sub_group.get_local_linear_id() == 0) {
                  bool check = (r_begin + r_acc.size() == r_end);
                  I scan = I(4);
                  for (int i = 0; i < size; ++i) {
                    check &= (U(scan) == r_begin[i]);
                    scan = op_max(scan, v_begin[i]);
                  }
                  res_acc[5] = check;
                }

                ASSERT_RETURN_TYPE(
                    U*,
                    sycl::joint_inclusive_scan(sub_group, v_begin, v_end,
                                               r_begin, op_plus, I(1412)),
                    "Return type of joint_inclusive_scan(sub_group g, InPtr "
                    "first, InPtr last, OutPtr result, BinaryOperation "
                    "binary_op, T init) is wrong\n");

                r_end = sycl::joint_inclusive_scan(sub_group, v_begin, v_end,
                                                   r_begin, op_plus, I(1412));
                if (sub_group.get_local_linear_id() == 0) {
                  bool check = (r_begin + r_acc.size() == r_end);
                  I scan = I(1412);
                  for (int i = 0; i < size; ++i) {
                    scan = op_plus(scan, v_begin[i]);
                    check &= (U(scan) == r_begin[i]);
                  }
                  res_acc[6] = check;
                }

                r_end = sycl::joint_inclusive_scan(sub_group, v_begin, v_end,
                                                   r_begin, op_max, I(4));
                if (sub_group.get_local_linear_id() == 0) {
                  bool check = (r_begin + r_acc.size() == r_end);
                  I scan = I(4);
                  for (int i = 0; i < size; ++i) {
                    scan = op_max(scan, v_begin[i]);
                    check &= (U(scan) == r_begin[i]);
                  }
                  res_acc[7] = check;
                }
              }
            });
      });
    }
    int index = 0;
    for (int i = 0; i < test_matrix; ++i)
      for (int j = 0; j < test_cases; ++j) {
        std::string work_group = util::work_group_print(work_group_range);
        CAPTURE(D, work_group, size);
        INFO("Value of " << test_names[i] << " with " << test_cases_names[j]
                         << " operation"
                            " and In = "
                         << type_name<T>() << ", Out = " << type_name<U>()
                         << ", T = " << type_name<I>() << " is "
                         << (res[index] ? "right" : "wrong"));
        CHECK(res[index++]);
      }
  }
}

template <typename DimensionT, typename T, typename U, typename I>
class invoke_init_joint_scan_group {
  static constexpr int D = DimensionT::value;

 public:
  void operator()(sycl::queue& queue) {
    init_joint_scan_group<D, T, U, I>(queue);
  }
};

// FIXME: Helper for implementations that cannot handle cases of different types
template <typename DimensionT, typename T>
class invoke_init_joint_scan_group_same_type {
  static constexpr int D = DimensionT::value;

 public:
  void operator()(sycl::queue& queue) {
    init_joint_scan_group<D, T, T, T>(queue);
  }
};

template <int D, typename T>
class scan_over_group_kernel;

/**
 * @brief Provides test for scans over group values
 * @tparam D Dimension to use for group instance
 * @tparam T Type used for value
 */
template <int D, typename T>
void scan_over_group(sycl::queue& queue) {
  // 4 functions * 2 function objects
  constexpr int test_matrix = 4;
  const std::string test_names[test_matrix] = {
      "T exclusive_scan_over_group(group g, T x, BinaryOperation binary_op)",
      "T inclusive_scan_over_group(group g, T x, BinaryOperation binary_op)",
      "T exclusive_scan_over_group(sub_group g, T x, BinaryOperation "
      "binary_op)",
      "T inclusive_scan_over_group(sub_group g, T x, BinaryOperation "
      "binary_op)"};
  constexpr int test_cases = 2;
  const std::string test_cases_names[test_cases] = {"plus", "maximum"};

  sycl::range<D> work_group_range = util::work_group_range<D>(queue);
  size_t work_group_size = work_group_range.size();

  // array to return results:
  std::valarray<bool> res(false, test_matrix * test_cases * work_group_size);
  {
    sycl::buffer<bool, 1> res_sycl(
        std::begin(res),
        sycl::range<1>(test_matrix * test_cases * work_group_size));

    queue.submit([&](sycl::handler& cgh) {
      auto res_acc = res_sycl.get_access<sycl::access::mode::read_write>(cgh);

      sycl::nd_range<D> executionRange(work_group_range, work_group_range);

      cgh.parallel_for<scan_over_group_kernel<D, T>>(
          executionRange, [=](sycl::nd_item<D> item) {
            T local_var;
            T scan_result;
            // checks are plagued by UB for too short types
            // so that the guards are introduced as the second parts in
            // local_res calculations
            size_t llid;
            bool local_res;

            auto op_plus = sycl::plus<T>();
            auto op_max = sycl::maximum<T>();

            sycl::group<D> group = item.get_group();

            llid = item.get_local_linear_id();
            local_var = T(llid + 1);

            ASSERT_RETURN_TYPE(
                T, sycl::exclusive_scan_over_group(group, local_var, op_plus),
                "Return type of exclusive_scan_over_group(group g, T x, "
                "BinaryOperation binary_op) is wrong\n");

            scan_result =
                sycl::exclusive_scan_over_group(group, local_var, op_plus);
            local_res = (scan_result == local_var * (local_var - 1) / 2) ||
                        (llid * (llid + 1) / 2 > util::exact_max<T>);
            res_acc[0 * work_group_size + llid] = local_res;

            scan_result =
                sycl::exclusive_scan_over_group(group, local_var, op_max);
            local_res =
                (scan_result == (llid == 0 ? std::numeric_limits<T>::lowest()
                                           : local_var - 1)) ||
                (llid + 1 > util::exact_max<T>);
            res_acc[1 * work_group_size + llid] = local_res;

            ASSERT_RETURN_TYPE(
                T, sycl::inclusive_scan_over_group(group, local_var, op_max),
                "Return type of inclusive_scan_over_group(group g, T x, "
                "BinaryOperation binary_op) is wrong\n");

            scan_result =
                sycl::inclusive_scan_over_group(group, local_var, op_plus);
            local_res = (scan_result == local_var * (local_var + 1) / 2) ||
                        ((llid + 1) * (llid + 2) / 2 > util::exact_max<T>);
            res_acc[2 * work_group_size + llid] = local_res;

            scan_result =
                sycl::inclusive_scan_over_group(group, local_var, op_max);
            local_res =
                (scan_result == local_var) || (llid + 1 > util::exact_max<T>);
            res_acc[3 * work_group_size + llid] = local_res;

            sycl::sub_group sub_group = item.get_sub_group();

            llid = sub_group.get_local_linear_id();
            local_var = T(llid + 1);

            ASSERT_RETURN_TYPE(
                T,
                sycl::exclusive_scan_over_group(sub_group, local_var, op_plus),
                "Return type of exclusive_scan_over_group(sub_group g, T x, "
                "BinaryOperation binary_op) is wrong\n");

            scan_result =
                sycl::exclusive_scan_over_group(sub_group, local_var, op_plus);
            local_res = (scan_result == local_var * (local_var - 1) / 2) ||
                        (llid * (llid + 1) / 2 > util::exact_max<T>);
            res_acc[4 * work_group_size + item.get_local_linear_id()] =
                local_res;

            scan_result =
                sycl::exclusive_scan_over_group(sub_group, local_var, op_max);
            local_res =
                (scan_result == (llid == 0 ? std::numeric_limits<T>::lowest()
                                           : local_var - 1)) ||
                (llid + 1 > util::exact_max<T>);
            res_acc[5 * work_group_size + item.get_local_linear_id()] =
                local_res;

            ASSERT_RETURN_TYPE(
                T,
                sycl::inclusive_scan_over_group(sub_group, local_var, op_max),
                "Return type of inclusive_scan_over_group(sub_group g, T x, "
                "BinaryOperation binary_op) is wrong\n");

            scan_result =
                sycl::inclusive_scan_over_group(sub_group, local_var, op_plus);
            local_res = (scan_result == local_var * (local_var + 1) / 2) ||
                        ((llid + 1) * (llid + 2) / 2 > util::exact_max<T>);
            res_acc[6 * work_group_size + item.get_local_linear_id()] =
                local_res;

            scan_result =
                sycl::inclusive_scan_over_group(sub_group, local_var, op_max);
            local_res =
                (scan_result == local_var) || (llid + 1 > util::exact_max<T>);
            res_acc[7 * work_group_size + item.get_local_linear_id()] =
                local_res;
          });
    });
  }
  int index = 0;
  for (int i = 0; i < test_matrix; ++i)
    for (int j = 0; j < test_cases; ++j) {
      bool result = res[index * work_group_size];
      for (size_t k = 1; k < work_group_size; ++k)
        result &= res[index * work_group_size + k];

      std::string work_group = util::work_group_print(work_group_range);
      CAPTURE(D, work_group);
      INFO("Value of " << test_names[i] << " with " << test_cases_names[j]
                       << " operation"
                          " and T = "
                       << type_name<T>() << " is "
                       << (result ? "right" : "wrong"));
      CHECK(result);
      ++index;
    }
}

template <typename DimensionT, typename T>
class invoke_scan_over_group {
  static constexpr int D = DimensionT::value;

 public:
  void operator()(sycl::queue& queue) { scan_over_group<D, T>(queue); }
};

template <int D, typename T, typename U>
class init_scan_over_group_kernel;

// FIXME: hipSYCL has wrong arguments order: init and op are interchanged
#ifdef SYCL_CTS_COMPILING_WITH_HIPSYCL
template <typename Group, typename V, typename T, typename BinaryOperation>
T inclusive_scan_over_group_impl(Group g, V x, BinaryOperation binary_op,
                                 T init) {
  return sycl::inclusive_scan_over_group(g, x, init, binary_op);
}
#else
template <typename Group, typename V, typename T, typename BinaryOperation>
T inclusive_scan_over_group_impl(Group g, V x, BinaryOperation binary_op,
                                 T init) {
  return sycl::inclusive_scan_over_group(g, x, binary_op, init);
}
#endif

// many errors with short types for hipSYCL
// it means conversion and calculation patterns are not OK
/**
 * @brief Provides test for scans over group with an init value
 * @tparam D Dimension to use for group instance
 * @tparam T Type used for init value and result
 * @tparam U Type used for group values
 */
template <int D, typename T, typename U>
void init_scan_over_group(sycl::queue& queue) {
  // 4 functions * 2 function objects
  constexpr int test_matrix = 4;
  const std::string test_names[test_matrix] = {
      "T exclusive_scan_over_group(group g, V x, T init, BinaryOperation "
      "binary_op)",
      "T inclusive_scan_over_group(group g, V x, BinaryOperation binary_op, T "
      "init)",
      "T exclusive_scan_over_group(sub_group g, V x, T init, BinaryOperation "
      "binary_op)",
      "T inclusive_scan_over_group(sub_group g, V x, BinaryOperation "
      "binary_op, T init)"};
  constexpr int test_cases = 2;
  const std::string test_cases_names[test_cases] = {"plus", "maximum"};

  sycl::range<D> work_group_range = util::work_group_range<D>(queue);
  size_t work_group_size = work_group_range.size();

  // array to return results:
  std::valarray<bool> res(false, test_matrix * test_cases * work_group_size);
  {
    sycl::buffer<bool, 1> res_sycl(
        std::begin(res),
        sycl::range<1>(test_matrix * test_cases * work_group_size));

    queue.submit([&](sycl::handler& cgh) {
      auto res_acc = res_sycl.get_access<sycl::access::mode::read_write>(cgh);

      sycl::nd_range<D> executionRange(work_group_range, work_group_range);

      cgh.parallel_for<init_scan_over_group_kernel<
          D, T, U>>(executionRange, [=](sycl::nd_item<D> item) {
        T init;
        U local_var;
        // checks are plagued by UB for too short types
        // so that the guards are introduced as the second parts in
        // local_res calculations
        size_t llid;
        size_t scan_result;
        bool local_res;

        auto op_plus = sycl::plus<T>();
        auto op_max = sycl::maximum<T>();

        sycl::group<D> group = item.get_group();

        llid = group.get_local_linear_id();
        local_var = U(llid + 1);

        ASSERT_RETURN_TYPE(
            T, sycl::exclusive_scan_over_group(group, local_var, init, op_plus),
            "Return type of exclusive_scan_over_group(group g, V x, T "
            "init, BinaryOperation binary_op) is wrong\n");

        // hipSYCL converts init from T to V
        init = T(1412);

        scan_result = llid * (llid + 1) / 2 + init;
        local_res = (scan_result == sycl::exclusive_scan_over_group(
                                        group, local_var, init, op_plus)) ||
                    (scan_result > util::exact_max<T>) ||
                    (llid > util::exact_max<U>);
        res_acc[0 * work_group_size + llid] = local_res;

        init = T(42);

        scan_result = (llid > init ? llid : static_cast<size_t>(init));
        local_res = (scan_result == sycl::exclusive_scan_over_group(
                                        group, local_var, init, op_max)) ||
                    (scan_result > util::exact_max<T>) ||
                    (llid > util::exact_max<U>);
        res_acc[1 * work_group_size + llid] = local_res;

        ASSERT_RETURN_TYPE(
            T, inclusive_scan_over_group_impl(group, local_var, op_max, init),
            "Return type of inclusive_scan_over_group(group g, V x, "
            "BinaryOperation binary_op, T init) is wrong\n");

        init = T(1412);

        scan_result = (llid + 1) * (llid + 2) / 2 + init;
        local_res = (scan_result == inclusive_scan_over_group_impl(
                                        group, local_var, op_plus, init)) ||
                    (scan_result > util::exact_max<T>) ||
                    (llid + 1 > util::exact_max<U>);
        res_acc[2 * work_group_size + llid] = local_res;

        init = T(42);

        scan_result = (llid + 1 > init ? llid + 1 : static_cast<size_t>(init));
        local_res = (scan_result == inclusive_scan_over_group_impl(
                                        group, local_var, op_max, init)) ||
                    (scan_result > util::exact_max<T>) ||
                    (llid + 1 > util::exact_max<U>);
        res_acc[3 * work_group_size + llid] = local_res;

        sycl::sub_group sub_group = item.get_sub_group();

        llid = sub_group.get_local_linear_id();
        local_var = U(llid + 1);

        ASSERT_RETURN_TYPE(
            T,
            sycl::exclusive_scan_over_group(sub_group, local_var, init,
                                            op_plus),
            "Return type of exclusive_scan_over_group(sub_group g, V x, T "
            "init, BinaryOperation binary_op) is wrong\n");

        // hipSYCL converts init from T to V=uint8_t, and calculates in V,
        // not in T, for T=floats, V=int hipSYCL uses 0 instead of init!
        init = T(1412);

        scan_result = llid * (llid + 1) / 2 + init;
        local_res = (scan_result == sycl::exclusive_scan_over_group(
                                        sub_group, local_var, init, op_plus)) ||
                    (scan_result > util::exact_max<T>) ||
                    (llid > util::exact_max<U>);
        res_acc[4 * work_group_size + item.get_local_linear_id()] = local_res;

        init = T(4);

        scan_result = (llid > init ? llid : static_cast<size_t>(init));
        local_res = (scan_result == sycl::exclusive_scan_over_group(
                                        sub_group, local_var, init, op_max)) ||
                    (scan_result > util::exact_max<T>) ||
                    (llid > util::exact_max<U>);
        res_acc[5 * work_group_size + item.get_local_linear_id()] = local_res;

        ASSERT_RETURN_TYPE(
            T,
            inclusive_scan_over_group_impl(sub_group, local_var, op_max, init),
            "Return type of inclusive_scan_over_group(sub_group g, V x, "
            "BinaryOperation binary_op, T init) is wrong\n");

        init = T(1412);

        scan_result = (llid + 1) * (llid + 2) / 2 + init;
        local_res = (scan_result == inclusive_scan_over_group_impl(
                                        sub_group, local_var, op_plus, init)) ||
                    (scan_result > util::exact_max<T>) ||
                    (llid + 1 > util::exact_max<U>);
        res_acc[6 * work_group_size + item.get_local_linear_id()] = local_res;

        init = T(4);

        scan_result = (llid + 1 > init ? llid + 1 : static_cast<size_t>(init));
        local_res = (scan_result == inclusive_scan_over_group_impl(
                                        sub_group, local_var, op_max, init)) ||
                    (scan_result > util::exact_max<T>) ||
                    (llid + 1 > util::exact_max<U>);
        res_acc[7 * work_group_size + item.get_local_linear_id()] = local_res;
      });
    });
  }
  int index = 0;
  for (int i = 0; i < test_matrix; ++i)
    for (int j = 0; j < test_cases; ++j) {
      bool result = res[index * work_group_size];
      for (size_t k = 1; k < work_group_size; ++k)
        result &= res[index * work_group_size + k];

      std::string work_group = util::work_group_print(work_group_range);
      CAPTURE(D, work_group);
      INFO("Value of " << test_names[i] << " with " << test_cases_names[j]
                       << " operation"
                          " and T = "
                       << type_name<T>() << ", V = " << type_name<U>() << " is "
                       << (result ? "right" : "wrong"));
      CHECK(result);
      ++index;
    }
}

template <typename DimensionT, typename T, typename U>
class invoke_init_scan_over_group {
  static constexpr int D = DimensionT::value;

 public:
  void operator()(sycl::queue& queue) { init_scan_over_group<D, T, U>(queue); }
};

// FIXME: Helper for implementations that cannot handle cases of different types
template <typename DimensionT, typename T>
class invoke_init_scan_over_group_same_type {
  static constexpr int D = DimensionT::value;

 public:
  void operator()(sycl::queue& queue) { init_scan_over_group<D, T, T>(queue); }
};

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
#include <optional>

constexpr size_t init = 1412;
static const auto Dims = integer_pack<1, 2, 3>::generate_unnamed();

template <typename T>
inline auto get_op_types() {
#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
  static const auto types =
      named_type_pack<sycl::plus<T>, sycl::multiplies<T>, sycl::bit_and<T>,
                      sycl::bit_or<T>, sycl::bit_xor<T>, sycl::logical_and<T>,
                      sycl::logical_or<T>, sycl::minimum<T>,
                      sycl::maximum<T>>::generate("plus", "multiplies",
                                                  "bit_and", "bit_or",
                                                  "bit_xor", "logical_and",
                                                  "logical_or", "minimum",
                                                  "maximum");
#else
  static const auto types =
      named_type_pack<sycl::plus<T>, sycl::maximum<T>>::generate("plus",
                                                                 "maximum");
#endif
  return types;
}

template <bool with_init, typename OpT, typename IteratorT>
size_t get_reduce_reference(IteratorT first, IteratorT end) {
  // Cast `init` to size_t so that guards are introduced in verification
  if constexpr (with_init)
    return std::accumulate(first, end, size_t(init), OpT());
  else
    return std::accumulate(first + 1, end, size_t(*first), OpT());
}

template <bool with_init, typename OpT, typename InputT, typename OutputT>
bool reduce_over_group_verify_helper(std::vector<InputT> &v_input, std::vector<OutputT> &v_output,
                                     size_t global_size, size_t local_size, size_t offset = 0) {
  bool res = false;
  const size_t count = (global_size + local_size - 1) / local_size;
  size_t beg = 0;
  for (size_t i = 0; i < count; ++i) {
    size_t cur_local_size = (i == count - 1 && global_size % local_size)
                                ? global_size % local_size
                                : local_size;
    auto v_input_begin = v_input.begin() + beg + offset;
    auto v_output_begin = v_output.begin() + beg + offset;
    const size_t group_reduced =
        get_reduce_reference<with_init, OpT>(v_input_begin, v_input_begin + cur_local_size);

    beg += cur_local_size;
    res = (group_reduced > util::exact_max<OutputT>) ||
          std::all_of(v_output_begin, v_output_begin + cur_local_size,
                      [=](OutputT i) { return i == group_reduced; });
    if (!res) break;
  }
  return res;
}

template <bool with_init, typename OpT, typename InputT, typename OutputT>
bool reduce_over_group_verifier(std::vector<InputT> &v_input, std::vector<OutputT> &v_output,
                                size_t global_size, size_t local_size) {
  return reduce_over_group_verify_helper<with_init, OpT>(
      v_input, v_output, global_size, local_size);
}

template <bool with_init, typename OpT, typename InputT, typename OutputT>
bool reduce_over_group_sg_verifier(std::vector<InputT> &v_input, std::vector<OutputT> &v_output, size_t global_size,
                                   size_t local_size, size_t sg_size) {
  bool res = false;
  const size_t count = (global_size + local_size - 1) / local_size;
  for (size_t i = 0; i < count; ++i) {
    res = reduce_over_group_verify_helper<with_init, OpT>(
        v_input, v_output, local_size, sg_size, local_size * i);
    if (!res) break;
  }
  return res;
}

template <int D, typename T, typename OpT>
class joint_reduce_group_kernel;

/**
 * @brief Provides test for joint reduce by group
 * @tparam D Dimension to use for group instance
 * @tparam T Type for reduced values
 * @tparam OpT Type for binary operator
 */
template <int D, typename T, typename OpT>
void joint_reduce_group(sycl::queue& queue, const std::string& op_name) {
  // 2 functions * 2 function objects
  constexpr int test_matrix = 2;
  const std::string test_names[test_matrix] = {
      "std::iterator_traits<Ptr>::value_type joint_reduce(group g, Ptr first, "
      "Ptr last, BinaryOperation binary_op)",
      "std::iterator_traits<Ptr>::value_type joint_reduce(sub_group g, Ptr "
      "first, Ptr last, BinaryOperation binary_op)"};

  sycl::range<D> work_group_range = sycl_cts::util::work_group_range<D>(queue);
  size_t work_group_size = work_group_range.size();

  const size_t sizes[3] = {5, work_group_size / 2, 3 * work_group_size};
  for (size_t size : sizes) {
    std::vector<T> v(size);
    std::iota(v.begin(), v.end(), 1);

    // checks are plagued by UB for too short types
    // so that the guards are introduced as the second parts in
    // res_acc calculations
    const size_t reduced = get_reduce_reference<false, OpT>(v.begin(), v.end());

    // array to return results
    bool res[test_matrix] = {false};
    {
      sycl::buffer<T, 1> v_sycl(v.data(), sycl::range<1>(size));
      sycl::buffer<bool, 1> res_sycl(res, sycl::range<1>(test_matrix));

      queue.submit([&](sycl::handler& cgh) {
        auto v_acc = v_sycl.template get_access<sycl::access::mode::read>(cgh);
        auto res_acc = res_sycl.get_access<sycl::access::mode::read_write>(cgh);

        sycl::nd_range<D> executionRange(work_group_range, work_group_range);

        cgh.parallel_for<joint_reduce_group_kernel<D, T, OpT>>(
            executionRange, [=](sycl::nd_item<D> item) {
              T* v_begin = v_acc.get_pointer();
              T* v_end = v_begin + v_acc.size();

              sycl::group<D> group = item.get_group();
              sycl::sub_group sub_group = item.get_sub_group();

              ASSERT_RETURN_TYPE(
                  T, sycl::joint_reduce(group, v_begin, v_end, OpT()),
                  "Return type of joint_reduce(group g, Ptr first, Ptr last, "
                  "BinaryOperation binary_op) is wrong\n");

              res_acc[0] = (reduced ==
                            sycl::joint_reduce(group, v_begin, v_end, OpT())) ||
                           (reduced > util::exact_max<T>);

              ASSERT_RETURN_TYPE(
                  T, sycl::joint_reduce(sub_group, v_begin, v_end, OpT()),
                  "Return type of joint_reduce(sub_group g, Ptr first, Ptr "
                  "last, BinaryOperation binary_op) is wrong\n");

          // FIXME: hipSYCL has no implementation over sub-groups
#ifdef SYCL_CTS_COMPILING_WITH_HIPSYCL
              res_acc[1] = true;
#else
              res_acc[1] = (reduced == sycl::joint_reduce(sub_group, v_begin, v_end, OpT()))
                || (reduced > util::exact_max<T>);
#endif
            });
      });
    }
    int index = 0;
    for (int i = 0; i < test_matrix; ++i) {
      std::string work_group =
          sycl_cts::util::work_group_print(work_group_range);
      CAPTURE(D, work_group, size);
      INFO("Value of " << test_names[i] << " with " << op_name
                       << " operation"
                          " and Ptr = "
                       << type_name<T>() << "* is "
                       << (res[index] ? "right" : "wrong"));
      CHECK(res[index++]);
    }
  }
}

template <typename DimensionT, typename T, typename OperatorT>
class invoke_joint_reduce_group {
  static constexpr int D = DimensionT::value;

 public:
  void operator()(sycl::queue& queue, const std::string& op_name) {
    joint_reduce_group<D, T, OperatorT>(queue, op_name);
  }
};

template <int D, typename T, typename U, typename OpT>
class init_joint_reduce_group_kernel;

/**
 * @brief Provides test for joint reduce by group with init
 * @tparam D Dimension to use for group instance
 * @tparam T Type for init and result values
 * @tparam U Type for reduced values
 * @tparam OpT Type for binary operator
 */
template <int D, typename T, typename U, typename OpT>
void init_joint_reduce_group(sycl::queue& queue, const std::string& op_name) {
  // 2 functions * 2 function objects
  constexpr int test_matrix = 2;
  const std::string test_names[test_matrix] = {
      "T joint_reduce(group g, Ptr first, Ptr last, T init, BinaryOperation "
      "binary_op)",
      "T joint_reduce(sub_group g, Ptr first, Ptr last, T init, "
      "BinaryOperation binary_op)"};

  sycl::range<D> work_group_range = sycl_cts::util::work_group_range<D>(queue);
  size_t work_group_size = work_group_range.size();

  const size_t sizes[3] = {5, work_group_size / 2, 3 * work_group_size};
  for (size_t size : sizes) {
    std::vector<U> v(size);
    std::iota(v.begin(), v.end(), 1);

    // checks are plagued by UB for too short types
    // so that the guards are introduced as the second parts in
    // res_acc calculations
    const size_t reduced = get_reduce_reference<true, OpT>(v.begin(), v.end());

    // array to return results
    bool res[test_matrix] = {false};
    {
      sycl::buffer<U, 1> v_sycl(v.data(), sycl::range<1>(size));
      sycl::buffer<bool, 1> res_sycl(res, sycl::range<1>(test_matrix));

      queue.submit([&](sycl::handler& cgh) {
        auto v_acc = v_sycl.template get_access<sycl::access::mode::read>(cgh);
        auto res_acc = res_sycl.get_access<sycl::access::mode::read_write>(cgh);

        sycl::nd_range<D> executionRange(work_group_range, work_group_range);

        cgh.parallel_for<init_joint_reduce_group_kernel<D, T, U, OpT>>(
            executionRange, [=](sycl::nd_item<D> item) {
              sycl::group<D> group = item.get_group();
              sycl::sub_group sub_group = item.get_sub_group();

              U* v_begin = v_acc.get_pointer();
              U* v_end = v_begin + v_acc.size();

              ASSERT_RETURN_TYPE(
                  T, sycl::joint_reduce(group, v_begin, v_end, T(init), OpT()),
                  "Return type of joint_reduce(group g, Ptr first, Ptr last, T "
                  "init, BinaryOperation binary_op) is wrong\n");

              res_acc[0] = (reduced == sycl::joint_reduce(group, v_begin, v_end,
                                                          T(init), OpT())) ||
                           (reduced > util::exact_max<T>) ||
                           (size > util::exact_max<U>);

              ASSERT_RETURN_TYPE(
                  T,
                  sycl::joint_reduce(sub_group, v_begin, v_end, T(init), OpT()),
                  "Return type of joint_reduce(sub_group g, Ptr first, Ptr "
                  "last, T init, BinaryOperation binary_op) is wrong\n");

          // FIXME: hipSYCL has no implementation over sub-groups
#ifdef SYCL_CTS_COMPILING_WITH_HIPSYCL
              res_acc[1] = true;
#else
              res_acc[1] = (reduced == sycl::joint_reduce(sub_group, v_begin, v_end, T(init), OpT()))
                || (reduced > util::exact_max<T>) || (size > util::exact_max<U>);
#endif
            });
      });
    }
    int index = 0;
    for (int i = 0; i < test_matrix; ++i) {
      std::string work_group =
          sycl_cts::util::work_group_print(work_group_range);
      CAPTURE(D, work_group, size);
      INFO("Value of " << test_names[i] << " with " << op_name
                       << " operation"
                          " and Ptr = "
                       << type_name<U>() << "*, T = " << type_name<T>()
                       << " is " << (res[index] ? "right" : "wrong"));
      CHECK(res[index++]);
    }
  }
}

template <typename DimensionT, typename RetT, typename ReducedT,
          typename OperatorT>
class invoke_init_joint_reduce_group {
  static constexpr int D = DimensionT::value;

 public:
  void operator()(sycl::queue& queue, const std::string& op_name) {
    init_joint_reduce_group<D, RetT, ReducedT, OperatorT>(queue, op_name);
  }
};

template <int D, typename T, typename OpT>
class reduce_over_group_kernel;

/**
 * @brief Provides test for reduce over group values
 * @tparam D Dimension to use for group instance
 * @tparam T Type for reduced values
 * @tparam OpT Type for binary operator
 */
template <int D, typename T, typename OpT>
void reduce_over_group(sycl::queue& queue, const std::string& op_name) {
  // 2 function * 2 function objects
  constexpr int test_matrix = 2;
  const std::string test_names[test_matrix] = {
      "T reduce_over_group(group g, T x, BinaryOperation binary_op)",
      "T reduce_over_group(sub_group g, T x, BinaryOperation binary_op)"};

  sycl::range<D> work_group_range = sycl_cts::util::work_group_range<D>(queue);
  size_t work_group_size = work_group_range.size();

  bool res = false;
  // array to input data
  std::vector<T> v(work_group_size);
  std::iota(v.begin(), v.end(), 1);
  // array to reduce results
  std::vector<T> group_output(work_group_size, 0);
  std::vector<T> sg_output(work_group_size, 0);

  // Store subgroup size
  size_t sg_size = 0;
  {
    sycl::buffer<T, 1> v_sycl(
        v.data(), sycl::range<1>(work_group_size));
    sycl::buffer<T, 1> g_output_sycl(
        group_output.data(), sycl::range<1>(work_group_size));
    sycl::buffer<T, 1> sg_output_sycl(
        sg_output.data(), sycl::range<1>(work_group_size));
    sycl::buffer<size_t> sgs_sycl(&sg_size, sycl::range<1>(1));

    queue.submit([&](sycl::handler& cgh) {
      auto v_acc = v_sycl.template get_access<sycl::access::mode::read>(cgh);
      auto g_output_acc =
          g_output_sycl.template get_access<sycl::access::mode::read_write>(cgh);
      auto sg_output_acc =
          sg_output_sycl.template get_access<sycl::access::mode::read_write>(cgh);
      auto sgs_acc = sgs_sycl.get_access<sycl::access::mode::read_write>(cgh);
      sycl::nd_range<D> executionRange(work_group_range, work_group_range);

      cgh.parallel_for<reduce_over_group_kernel<D, T, OpT>>(
          executionRange, [=](sycl::nd_item<D> item) {
            sycl::group<D> group = item.get_group();
            size_t group_size = group.get_local_linear_range();
            size_t index = item.get_global_linear_id();

            ASSERT_RETURN_TYPE(T,
                               sycl::reduce_over_group(group, v_acc[index], OpT()),
                               "Return type of reduce_over_group(group g, T x, "
                               "BinaryOperation binary_op) is wrong\n");

            g_output_acc[index] =
                sycl::reduce_over_group(group, v_acc[index], OpT());

            sycl::sub_group sub_group = item.get_sub_group();
            sgs_acc[0] = sub_group.get_local_linear_range();

            ASSERT_RETURN_TYPE(
                T, sycl::reduce_over_group(sub_group, v_acc[index], OpT()),
                "Return type of reduce_over_group(sub_group g, "
                "T x, BinaryOperation binary_op) is wrong\n");
            sg_output_acc[index] =
                sycl::reduce_over_group(sub_group, v_acc[index], OpT());
          });
    });
  }

  // // Verify return value for reduce_over_group on group
  {
    res = reduce_over_group_verifier<false, OpT>(
        v, group_output, work_group_size, work_group_size);
    std::string work_group = sycl_cts::util::work_group_print(work_group_range);
    CAPTURE(D, work_group);
    INFO("Value of " << test_names[0] << " with " << op_name
                     << " operation and T = " << type_name<T>() << " over group"
                     << " is " << (res ? "right" : "wrong"));
    CHECK(res);
  }

  // Verify return value for reduce_over_group on sub_group
  {
    res = reduce_over_group_sg_verifier<false, OpT>(
        v, sg_output, work_group_size, work_group_size,
        sg_size);
    std::string work_group = sycl_cts::util::work_group_print(work_group_range);
    CAPTURE(D, work_group);
    INFO("Value of " << test_names[0] << " with " << op_name
                     << " operation and T = " << type_name<T>()
                     << " over sub_group"
                     << " is " << (res ? "right" : "wrong"));
    CHECK(res);
  }
}

template <typename DimensionT, typename T, typename OperatorT>
class invoke_reduce_over_group {
  static constexpr int D = DimensionT::value;

 public:
  void operator()(sycl::queue& queue, const std::string& op_name) {
    reduce_over_group<D, T, OperatorT>(queue, op_name);
  }
};

template <int D, typename T, typename U, typename OpT>
class init_reduce_over_group_kernel;

/**
 * @brief Provides test for reduce over group values with init
 * @tparam D Dimension to use for group instance
 * @tparam T Type for group values
 * @tparam U Type for init and result values
 * @tparam OpT Type for binary operator
 */
template <int D, typename T, typename U, typename OpT>
void init_reduce_over_group(sycl::queue& queue, const std::string& op_name) {
  // 2 function * 2 function objects
  constexpr int test_matrix = 2;
  const std::string test_names[test_matrix] = {
      "T reduce_over_group(group g, V x, T init, BinaryOperation binary_op)",
      "T reduce_over_group(sub_group g, V x, T init, BinaryOperation "
      "binary_op)"};

  sycl::range<D> work_group_range = sycl_cts::util::work_group_range<D>(queue);
  size_t work_group_size = work_group_range.size();

  bool res = false;
  // array to input data
  std::vector<T> v(work_group_size);
  std::iota(v.begin(), v.end(), 1);
  // array to reduce results
  std::vector<T> group_output(work_group_size, 0);
  std::vector<T> sg_output(work_group_size, 0);

  // Store subgroup size
  size_t sg_size = 0;
  {
    sycl::buffer<T, 1> v_sycl(
        v.data(), sycl::range<1>(work_group_size));
    sycl::buffer<T, 1> g_output_sycl(
        group_output.data(), sycl::range<1>(work_group_size));
    sycl::buffer<T, 1> sg_output_sycl(
        sg_output.data(), sycl::range<1>(work_group_size));
    sycl::buffer<size_t> sgs_sycl(&sg_size, sycl::range<1>(1));

    queue.submit([&](sycl::handler& cgh) {
      auto v_acc = v_sycl.template get_access<sycl::access::mode::read>(cgh);
      auto g_output_acc =
          g_output_sycl.template get_access<sycl::access::mode::read_write>(cgh);
      auto sg_output_acc =
          sg_output_sycl.template get_access<sycl::access::mode::read_write>(cgh);
      auto sgs_acc = sgs_sycl.get_access<sycl::access::mode::read_write>(cgh);
      sycl::nd_range<D> executionRange(work_group_range, work_group_range);

      cgh.parallel_for<init_reduce_over_group_kernel<D, T, U, OpT>>(
          executionRange, [=](sycl::nd_item<D> item) {
            sycl::group<D> group = item.get_group();
            size_t group_size = group.get_local_linear_range();
            size_t index = item.get_global_linear_id();

            ASSERT_RETURN_TYPE(T,
                               sycl::reduce_over_group(
                                   group, v_acc[index], T(init), OpT()),
                               "Return type of reduce_over_group(group g, V x, "
                               "T init, BinaryOperation binary_op) is wrong\n");

            g_output_acc[index] =
                sycl::reduce_over_group(group, v_acc[index], T(init), OpT());

            sycl::sub_group sub_group = item.get_sub_group();
            sgs_acc[0] = sub_group.get_local_linear_range();

            ASSERT_RETURN_TYPE(
                T,
                sycl::reduce_over_group(sub_group, v_acc[index], T(init),
                                        OpT()),
                "Return type of reduce_over_group(sub_group g, V x, T init, "
                "BinaryOperation binary_op) is wrong\n");
            sg_output_acc[index] =
                sycl::reduce_over_group(sub_group, v_acc[index], T(init), OpT());
          });
    });
  }

  // // Verify return value for reduce_over_group on group
  {
    res = reduce_over_group_verifier<true, OpT>(
        v, group_output, work_group_size, work_group_size);
    std::string work_group = sycl_cts::util::work_group_print(work_group_range);
    CAPTURE(D, work_group);
    INFO("Value of " << test_names[0] << " with " << op_name
                     << " operation and T = " << type_name<T>() << " over group"
                     << " is " << (res ? "right" : "wrong"));
    CHECK(res);
  }

  // Verify return value for reduce_over_group on sub_group
  {
    res = reduce_over_group_sg_verifier<true, OpT>(
        v, sg_output, work_group_size, work_group_size,
        sg_size);
    std::string work_group = sycl_cts::util::work_group_print(work_group_range);
    CAPTURE(D, work_group);
    INFO("Value of " << test_names[0] << " with " << op_name
                     << " operation and T = " << type_name<T>()
                     << " over sub_group"
                     << " is " << (res ? "right" : "wrong"));
    CHECK(res);
  }
}

template <typename DimensionT, typename RetT, typename ReducedT,
          typename OperatorT>
class invoke_init_reduce_over_group {
  static constexpr int D = DimensionT::value;

 public:
  void operator()(sycl::queue& queue, const std::string& op_name) {
    init_reduce_over_group<D, RetT, ReducedT, OperatorT>(queue, op_name);
  }
};

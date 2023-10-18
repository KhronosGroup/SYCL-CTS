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

template <int D, typename T, typename U, typename I, typename OpT>
class joint_scan_group_kernel;

constexpr int init = 42;
constexpr size_t test_size = 12;

template <typename I, typename T, typename U, typename Group, typename OpT>
auto joint_inclusive_scan_helper(Group group, T* v_begin, T* v_end,
                                 U* r_i_begin, OpT op, bool with_init) {
  if (with_init) {
    return sycl::joint_inclusive_scan(group, v_begin, v_end, r_i_begin, op,
                                      I(init));
  }
  assert((std::is_same_v<I, U> &&
          "Without init value I and U should be the same type."));
  return (U*)sycl::joint_inclusive_scan(group, v_begin, v_end, (I*)r_i_begin,
                                        op);
}

template <typename I, typename T, typename U, typename Group, typename OpT>
auto joint_exclusive_scan_helper(Group group, T* v_begin, T* v_end,
                                 U* r_e_begin, OpT op, bool with_init) {
  if (with_init) {
    return sycl::joint_exclusive_scan(group, v_begin, v_end, r_e_begin, I(init),
                                      op);
  }
  assert((std::is_same_v<I, U> &&
          "Without init value I and U should be the same type."));
  return (U*)sycl::joint_exclusive_scan(group, v_begin, v_end, (I*)r_e_begin,
                                        op);
}

template <typename T, typename U>
struct JointScanDataStruct {
  JointScanDataStruct(size_t range_size)
      : ref_input(range_size), res(range_size * 4, U(-1)) {
    std::iota(ref_input.begin(), ref_input.end(), T(1));
  }

  template <typename I, typename OpT>
  void check_results(size_t range_size, OpT op, const std::string& op_name,
                     bool with_init) {
    CHECK(end[0]);
    CHECK(end[1]);
    CHECK(end[2]);
    CHECK(end[3]);
    CHECK(ret_type[0]);
    CHECK(ret_type[1]);
    CHECK(ret_type[2]);
    CHECK(ret_type[3]);

    I init_value = with_init ? I(init) : sycl::known_identity<OpT, I>::value;

    std::vector<U> reference_e(range_size, U(-1));
    std::vector<U> reference_i(range_size, U(-1));
    std::exclusive_scan(ref_input.begin(), ref_input.end(), reference_e.begin(),
                        init_value, op);
    std::inclusive_scan(ref_input.begin(), ref_input.end(), reference_i.begin(),
                        op, init_value);
    // res consists of 4 series of results: two pairs of exclusive and inclusive
    // scan results made over 'group' and 'sub_group' accordingly.
    for (int group_i = 0; group_i < 2; group_i++) {
      std::string group_name = group_i == 0 ? "group" : "sub_group";
      size_t group_offset = range_size * group_i;
      for (int i = 0; i < range_size; i++) {
        // Each group contains two sets of results.
        size_t res_i = i + 2 * group_offset;
        {
          INFO("Check joint_exclusive_scan on " + group_name + " for element " +
               std::to_string(i) + " (Operator: " + op_name + ")");
          INFO("Result: " + std::to_string(res[res_i]));
          INFO("Expected: " + std::to_string(reference_e[i]));
          CHECK(res[res_i] == reference_e[i]);
        }
        {
          INFO("Check joint_inclusive_scan on " + group_name + " for element " +
               std::to_string(i) + " (Operator: " + op_name + ")");
          INFO("Result: " + std::to_string(res[res_i + range_size]));
          INFO("Expected: " + std::to_string(reference_i[i]));
          CHECK(res[res_i + range_size] == reference_i[i]);
        }
      }
    }
  }

  sycl::buffer<T, 1> create_ref_input_buffer() {
    return {ref_input.data(), ref_input.size()};
  }

  sycl::buffer<U, 1> create_res_buffer() { return {res.data(), res.size()}; }

  sycl::buffer<bool, 1> create_end_buffer() { return {end, 4}; }

  sycl::buffer<bool, 1> create_ret_type_buffer() { return {ret_type, 4}; }

  std::vector<T> ref_input;
  std::vector<U> res;
  bool end[4] = {false, false, false, false};
  bool ret_type[4] = {false, false, false, false};
  std::vector<size_t> local_id;
};

template <int D, typename T, typename U, typename I = U, typename OpT>
void check_scan(sycl::queue& queue, size_t size,
                sycl::nd_range<D> executionRange, OpT op,
                const std::string& op_name, bool with_init) {
  JointScanDataStruct<T, U> host_data{size};
  {
    sycl::buffer<T, 1> ref_input_sycl = host_data.create_ref_input_buffer();
    sycl::buffer<U, 1> res_sycl = host_data.create_res_buffer();
    sycl::buffer<bool, 1> end_sycl = host_data.create_end_buffer();
    sycl::buffer<bool, 1> ret_type_sycl = host_data.create_ret_type_buffer();

    queue
        .submit([&](sycl::handler& cgh) {
          sycl::accessor<T, 1> ref_input_acc(ref_input_sycl, cgh);
          sycl::accessor<U, 1> res_acc(res_sycl, cgh);
          sycl::accessor<bool, 1> end_acc(end_sycl, cgh);
          sycl::accessor<bool, 1> ret_type_acc(ret_type_sycl, cgh);

          cgh.parallel_for<joint_scan_group_kernel<D, T, U, I, OpT>>(
              executionRange, [=](sycl::nd_item<D> item) {
                sycl::group<D> group = item.get_group();
                sycl::sub_group sub_group = item.get_sub_group();

                T* v_begin = ref_input_acc.get_pointer();
                T* v_end = v_begin + ref_input_acc.size();

                U* r_g_e_begin = res_acc.get_pointer();
                U* r_g_i_begin = res_acc.get_pointer() + size;
                U* r_sg_e_begin = res_acc.get_pointer() + size * 2;
                U* r_sg_i_begin = res_acc.get_pointer() + size * 3;

                auto r_g_e_end = joint_exclusive_scan_helper<I>(
                    group, v_begin, v_end, r_g_e_begin, op, with_init);
                ret_type_acc[0] = std::is_same_v<U*, decltype(r_g_e_end)>;

                auto r_g_i_end = joint_inclusive_scan_helper<I>(
                    group, v_begin, v_end, r_g_i_begin, op, with_init);
                ret_type_acc[1] = std::is_same_v<U*, decltype(r_g_i_end)>;

                auto r_sg_e_end = joint_exclusive_scan_helper<I>(
                    sub_group, v_begin, v_end, r_sg_e_begin, op, with_init);
                ret_type_acc[2] = std::is_same_v<U*, decltype(r_sg_e_end)>;

                auto r_sg_i_end = joint_inclusive_scan_helper<I>(
                    sub_group, v_begin, v_end, r_sg_i_begin, op, with_init);
                ret_type_acc[3] = std::is_same_v<U*, decltype(r_sg_i_end)>;

                end_acc[0] = (r_g_e_begin + size == r_g_e_end);
                end_acc[1] = (r_g_i_begin + size == r_g_i_end);
                end_acc[2] = (r_sg_e_begin + size == r_sg_e_end);
                end_acc[3] = (r_sg_i_begin + size == r_sg_i_end);
              });
        })
        .wait_and_throw();
  }

  host_data.template check_results<I>(size, op, op_name, with_init);
}

/**
 * @brief Provides test for joint scans
 * @tparam D Dimension to use for group instance
 * @tparam T Type pointed by InPtr
 * @tparam U Type pointed by OutPtr
 * @tparam OperatorT Type of binary operation
 */
template <typename DimensionT, typename T, typename U, typename OperatorT>
struct joint_scan_group {
  static constexpr int D = DimensionT::value;

  void operator()(sycl::queue& queue, const std::string& op_name) {
    if constexpr (type_traits::group_algorithms::is_legal_operator_v<
                      U, OperatorT>) {
      INFO(" with types " + type_name<T>() + " and " + type_name<U>());

      sycl::range<D> work_group_range =
          sycl_cts::util::work_group_range<D>(queue, test_size);

      size_t work_group_size = work_group_range.size();

      sycl::nd_range<D> executionRange(work_group_range, work_group_range);

      const size_t sizes[2] = {5, 2};
      for (size_t size : sizes) {
        check_scan<D, T, U>(queue, size, executionRange, OperatorT(), op_name,
                            false);
      }
    }
  }
};

template <typename DimensionT, typename T, typename U>
class invoke_joint_scan_group {
 public:
  void operator()(sycl::queue& queue) {
    if constexpr (std::is_same_v<std::remove_cv_t<T>, sycl::half>) {
      if (!queue.get_device().has(sycl::aspect::fp16))
        SKIP(
            "Device does not support half precision floating point "
            "operations.");
    } else if (std::is_same_v<std::remove_cv_t<T>, double>) {
      if (!queue.get_device().has(sycl::aspect::fp64))
        SKIP(
            "Device does not support double precision floating point "
            "operations.");
    }

    const auto operators = get_op_types<U>();
    for_all_combinations<joint_scan_group, DimensionT, T, U>(operators, queue);
  }
};

// FIXME: Helper for implementations that cannot handle cases of different types
template <typename DimensionT, typename T>
class invoke_joint_scan_group_same_type {
 public:
  void operator()(sycl::queue& queue) {
    if constexpr (std::is_same_v<std::remove_cv_t<T>, sycl::half>) {
      if (!queue.get_device().has(sycl::aspect::fp16))
        SKIP(
            "Device does not support half precision floating point "
            "operations.");
    } else if (std::is_same_v<std::remove_cv_t<T>, double>) {
      if (!queue.get_device().has(sycl::aspect::fp64))
        SKIP(
            "Device does not support double precision floating point "
            "operations.");
    }

    const auto operators = get_op_types<T>();
    for_all_combinations<joint_scan_group, DimensionT, T, T>(operators, queue);
  }
};

template <int D, typename T, typename U, typename I>
class init_joint_scan_group_kernel;

/**
 * @brief Provides test for joint scans with init
 * @tparam D Dimension to use for group instance
 * @tparam T Type pointed by InPtr
 * @tparam U Type pointed by OutPtr
 * @tparam I Type used for init value
 * @tparam OperatorT Type of binary operation
 */
template <typename DimensionT, typename T, typename U, typename I,
          typename OperatorT>
struct init_joint_scan_group {
  static constexpr int D = DimensionT::value;

  void operator()(sycl::queue& queue, const std::string& op_name) {
    if constexpr (type_traits::group_algorithms::is_legal_operator_v<
                      I, OperatorT>) {
      INFO(" with types " + type_name<T>() + " and " + type_name<U>() +
           ", init type " + type_name<I>());

      sycl::range<D> work_group_range =
          sycl_cts::util::work_group_range<D>(queue, test_size);
      sycl::nd_range<D> executionRange(work_group_range, work_group_range);

      size_t work_group_size = work_group_range.size();

      const size_t sizes[2] = {5, 2};
      for (size_t size : sizes) {
        check_scan<D, T, U, I>(queue, size, executionRange, OperatorT(),
                               op_name, true);
      }
    }
  }
};

template <typename DimensionT, typename T, typename U, typename I>
class invoke_init_joint_scan_group {
 public:
  void operator()(sycl::queue& queue) {
    if constexpr (std::is_same_v<std::remove_cv_t<T>, sycl::half>) {
      if (!queue.get_device().has(sycl::aspect::fp16))
        SKIP(
            "Device does not support half precision floating point "
            "operations.");
    } else if (std::is_same_v<std::remove_cv_t<T>, double>) {
      if (!queue.get_device().has(sycl::aspect::fp64))
        SKIP(
            "Device does not support double precision floating point "
            "operations.");
    }

    const auto operators = get_op_types<I>();
    for_all_combinations<init_joint_scan_group, DimensionT, T, U, I>(operators,
                                                                     queue);
  }
};

// FIXME: Helper for implementations that cannot handle cases of different types
template <typename DimensionT, typename T>
class invoke_init_joint_scan_group_same_type {
 public:
  void operator()(sycl::queue& queue) {
    if constexpr (std::is_same_v<std::remove_cv_t<T>, sycl::half>) {
      if (!queue.get_device().has(sycl::aspect::fp16))
        SKIP(
            "Device does not support half precision floating point "
            "operations.");
    } else if (std::is_same_v<std::remove_cv_t<T>, double>) {
      if (!queue.get_device().has(sycl::aspect::fp64))
        SKIP(
            "Device does not support double precision floating point "
            "operations.");
    }

    const auto operators = get_op_types<T>();
    for_all_combinations<init_joint_scan_group, DimensionT, T, T, T>(operators,
                                                                     queue);
  }
};

template <int D, typename T, typename U, typename OpT>
class scan_over_group_kernel;

template <typename T, typename U, typename Group, typename OpT>
auto inclusive_scan_over_group_helper(Group group, U x, OpT op,
                                      bool with_init) {
  if (with_init) {
    return sycl::inclusive_scan_over_group(group, x, op, T(init));
  }
  assert((std::is_same_v<T, U> &&
          "Without init value T and U should be the same type."));
  return sycl::inclusive_scan_over_group(group, T(x), op);
}

template <typename T, typename U, typename Group, typename OpT>
auto exclusive_scan_over_group_helper(Group group, U x, OpT op,
                                      bool with_init) {
  if (with_init) {
    return sycl::exclusive_scan_over_group(group, x, T(init), op);
  }
  assert((std::is_same_v<T, U> &&
          "Without init value T and U should be the same type."));
  return sycl::exclusive_scan_over_group(group, T(x), op);
}

template <typename T, typename U>
struct ScanOverGroupDataStruct {
  ScanOverGroupDataStruct(size_t range_size)
      : ref_input(range_size),
        res(range_size * 4, T(-1)),
        local_id(range_size * 2, 0) {
    std::iota(ref_input.begin(), ref_input.end(), U(1));
  }

  template <typename OpT>
  void check_results(size_t range_size, OpT op, const std::string& op_name,
                     bool with_init) {
    CHECK(ret_type[0]);
    CHECK(ret_type[1]);
    CHECK(ret_type[2]);
    CHECK(ret_type[3]);

    T init_value = with_init ? T(init) : sycl::known_identity<OpT, T>::value;
    // res consists of 4 series of results: two pairs of exclusive and inclusive
    // scan results made over 'group' and 'sub_group' accordingly.
    for (int group_i = 0; group_i < 2; group_i++) {
      std::string group_name = group_i == 0 ? "group" : "sub_group";
      size_t group_offset = range_size * group_i;
      for (int i = 0; i < range_size; i++) {
        int shift = i - local_id[i + group_offset];
        auto startIter = ref_input.begin() + shift;
        // Each group contains two sets of results.
        size_t res_i = i + 2 * group_offset;
        {
          INFO("Check exclusive_scan_over_group on " + group_name +
               " for element " + std::to_string(i) + " (Operator: " + op_name +
               ")");
          std::vector<T> reference(i + 1, T(-1));
          std::exclusive_scan(startIter, ref_input.begin() + i + 1,
                              reference.begin(), init_value, op);
          INFO("Result: " + std::to_string(res[res_i]));
          INFO("Expected: " + std::to_string(reference[i - shift]));
          CHECK(res[res_i] == reference[i - shift]);
        }
        {
          INFO("Check inclusive_scan_over_group on " + group_name +
               " for element " + std::to_string(i) + " (Operator: " + op_name +
               ")");
          std::vector<T> reference(i + 1, T(-1));
          std::inclusive_scan(startIter, ref_input.begin() + i + 1,
                              reference.begin(), op, init_value);
          INFO("Result: " + std::to_string(res[res_i + range_size]));
          INFO("Expected: " + std::to_string(reference[i - shift]));
          CHECK(res[res_i + range_size] == reference[i - shift]);
        }
      }
    }
  }

  sycl::buffer<U, 1> create_ref_input_buffer() {
    return {ref_input.data(), ref_input.size()};
  }

  sycl::buffer<T, 1> create_res_buffer() { return {res.data(), res.size()}; }

  sycl::buffer<bool, 1> create_ret_type_buffer() { return {ret_type, 4}; }

  sycl::buffer<size_t, 1> create_local_id_buffer() {
    return {local_id.data(), local_id.size()};
  }

  std::vector<U> ref_input;
  std::vector<T> res;
  bool ret_type[4] = {false, false, false, false};
  std::vector<size_t> local_id;
};

template <int D, typename T, typename U = T, typename OpT>
void check_scan_over_group(sycl::queue& queue, sycl::range<D> range, OpT op,
                           const std::string& op_name, bool with_init) {
  auto range_size = range.size();
  REQUIRE(((range_size * (range_size + 1) / 2) + T(init)) <=
          std::numeric_limits<T>::max());

  ScanOverGroupDataStruct<T, U> host_data{range_size};
  {
    auto ref_input_sycl = host_data.create_ref_input_buffer();
    auto res_sycl = host_data.create_res_buffer();
    auto ret_type_sycl = host_data.create_ret_type_buffer();
    auto local_id_sycl = host_data.create_local_id_buffer();

    queue
        .submit([&](sycl::handler& cgh) {
          sycl::accessor<U, 1, sycl::access_mode::read> ref_input_acc(
              ref_input_sycl, cgh);
          sycl::accessor<T, 1> res_acc(res_sycl, cgh);
          sycl::accessor<bool, 1> ret_type_acc(ret_type_sycl, cgh);
          sycl::accessor<size_t, 1> local_id_acc(local_id_sycl, cgh);

          cgh.parallel_for<scan_over_group_kernel<D, T, U, OpT>>(
              sycl::nd_range<D>(range, range), [=](sycl::nd_item<D> item) {
                sycl::group<D> group = item.get_group();
                sycl::sub_group sub_group = item.get_sub_group();

                auto index = item.get_global_linear_id();
                local_id_acc[index] = group.get_local_linear_id();
                local_id_acc[range_size + index] =
                    sub_group.get_local_linear_id();

                auto res_g_e = exclusive_scan_over_group_helper<T>(
                    group, ref_input_acc[index], op, with_init);
                res_acc[index] = res_g_e;
                ret_type_acc[0] = std::is_same_v<T, decltype(res_g_e)>;

                auto res_g_i = inclusive_scan_over_group_helper<T>(
                    group, ref_input_acc[index], op, with_init);
                res_acc[range_size + index] = res_g_i;
                ret_type_acc[1] = std::is_same_v<T, decltype(res_g_i)>;

                auto res_sg_e = exclusive_scan_over_group_helper<T>(
                    sub_group, ref_input_acc[index], op, with_init);
                res_acc[range_size * 2 + index] = res_sg_e;
                ret_type_acc[2] = std::is_same_v<T, decltype(res_sg_e)>;

                auto res_sg_i = inclusive_scan_over_group_helper<T>(
                    sub_group, ref_input_acc[index], op, with_init);
                res_acc[range_size * 3 + index] = res_sg_i;
                ret_type_acc[3] = std::is_same_v<T, decltype(res_sg_i)>;
              });
        })
        .wait_and_throw();
  }

  host_data.check_results(range_size, op, op_name, with_init);
}

/**
 * @brief Provides test for scans over group values
 * @tparam D Dimension to use for group instance
 * @tparam T Type used for value
 * @tparam OperatorT Type of binary operation
 */
template <typename DimensionT, typename T, typename OperatorT>
struct scan_over_group {
  static constexpr int D = DimensionT::value;

  void operator()(sycl::queue& queue, const std::string& op_name) {
    if constexpr (type_traits::group_algorithms::is_legal_operator_v<
                      T, OperatorT>) {
      INFO(" with type " + type_name<T>());

      sycl::range<D> work_group_range =
          sycl_cts::util::work_group_range<D>(queue, test_size);
      size_t work_group_size = work_group_range.size();

      check_scan_over_group<D, T>(queue, work_group_range, OperatorT(), op_name,
                                  false);
    }
  }
};

template <typename DimensionT, typename T>
class invoke_scan_over_group {
 public:
  void operator()(sycl::queue& queue) {
    if constexpr (std::is_same_v<std::remove_cv_t<T>, sycl::half>) {
      if (!queue.get_device().has(sycl::aspect::fp16))
        SKIP(
            "Device does not support half precision floating point "
            "operations.");
    } else if (std::is_same_v<std::remove_cv_t<T>, double>) {
      if (!queue.get_device().has(sycl::aspect::fp64))
        SKIP(
            "Device does not support double precision floating point "
            "operations.");
    }

    const auto operators = get_op_types<T>();
    for_all_combinations<scan_over_group, DimensionT, T>(operators, queue);
  }
};

template <int D, typename T, typename U>
class init_scan_over_group_kernel;

// many errors with short types for hipSYCL
// it means conversion and calculation patterns are not OK
/**
 * @brief Provides test for scans over group with an init value
 * @tparam D Dimension to use for group instance
 * @tparam T Type used for init value and result
 * @tparam U Type used for group values
 * @tparam OperatorT Type of binary operation
 */
template <typename DimensionT, typename T, typename U, typename OperatorT>
struct init_scan_over_group {
  static constexpr int D = DimensionT::value;

  void operator()(sycl::queue& queue, const std::string& op_name) {
    if constexpr (type_traits::group_algorithms::is_legal_operator_v<
                      T, OperatorT>) {
      INFO(" with types " + type_name<T>() + " and " + type_name<U>());

      sycl::range<D> work_group_range =
          sycl_cts::util::work_group_range<D>(queue, test_size);

      check_scan_over_group<D, T, U>(queue, work_group_range, OperatorT(),
                                     op_name, true);
    }
  }
};

template <typename DimensionT, typename T, typename U>
class invoke_init_scan_over_group {
 public:
  void operator()(sycl::queue& queue) {
    if constexpr (std::is_same_v<std::remove_cv_t<T>, sycl::half>) {
      if (!queue.get_device().has(sycl::aspect::fp16))
        SKIP(
            "Device does not support half precision floating point "
            "operations.");
    } else if (std::is_same_v<std::remove_cv_t<T>, double>) {
      if (!queue.get_device().has(sycl::aspect::fp64))
        SKIP(
            "Device does not support double precision floating point "
            "operations.");
    }

    const auto operators = get_op_types<T>();
    for_all_combinations<init_scan_over_group, DimensionT, T, U>(operators,
                                                                 queue);
  }
};

// FIXME: Helper for implementations that cannot handle cases of different types
template <typename DimensionT, typename T>
class invoke_init_scan_over_group_same_type {
 public:
  void operator()(sycl::queue& queue) {
    const auto operators = get_op_types<T>();
    for_all_combinations<init_scan_over_group, DimensionT, T, T>(operators,
                                                                 queue);
  }
};

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

#include "../../util/accuracy.h"
#include "group_functions_common.h"

template <int D, typename T, typename U, typename Group, bool with_init,
          typename I, typename OpT>
class joint_scan_group_kernel;

constexpr int init = 42;

template <typename Group, int D>
Group get_group(const sycl::nd_item<D>& item) {
  if constexpr (std::is_same_v<std::decay_t<Group>, sycl::sub_group>)
    return item.get_sub_group();
  else
    return item.get_group();
}

template <bool with_init, typename I, typename T, typename U, typename Group,
          typename OpT>
auto joint_inclusive_scan_helper(Group group, T* v_begin, T* v_end,
                                 U* r_i_begin, OpT op) {
  if constexpr (with_init) {
    return sycl::joint_inclusive_scan(group, v_begin, v_end, r_i_begin, op,
                                      I(init));
  } else
    return sycl::joint_inclusive_scan(group, v_begin, v_end, r_i_begin, op);
}

template <bool with_init, typename I, typename T, typename U, typename Group,
          typename OpT>
auto joint_exclusive_scan_helper(Group group, T* v_begin, T* v_end,
                                 U* r_e_begin, OpT op) {
  if constexpr (with_init) {
    return sycl::joint_exclusive_scan(group, v_begin, v_end, r_e_begin, I(init),
                                      op);
  } else
    return sycl::joint_exclusive_scan(group, v_begin, v_end, r_e_begin, op);
}

template <int D, typename T, typename U, typename Group, bool with_init,
          typename I = U, typename OpT>
void check_scan(sycl::queue& queue, size_t size,
                sycl::nd_range<D> executionRange, OpT op) {
  std::vector<T> v(size);
  std::iota(v.begin(), v.end(), T(1));
  std::vector<U> res_e(size, U(-1));
  std::vector<U> res_i(size, U(-1));
  bool res_e_end = false;
  bool res_i_end = false;
  bool ret_type_e = false;
  bool ret_type_i = false;
  {
    sycl::buffer<T, 1> v_sycl(v.data(), sycl::range<1>(size));
    sycl::buffer<U, 1> res_e_sycl(res_e.data(), sycl::range<1>(size));
    sycl::buffer<U, 1> res_i_sycl(res_i.data(), sycl::range<1>(size));
    sycl::buffer<bool, 1> end_e_sycl(&res_e_end, sycl::range<1>(1));
    sycl::buffer<bool, 1> end_i_sycl(&res_i_end, sycl::range<1>(1));
    sycl::buffer<bool, 1> ret_type_e_sycl(&ret_type_e, sycl::range<1>(1));
    sycl::buffer<bool, 1> ret_type_i_sycl(&ret_type_i, sycl::range<1>(1));

    queue
        .submit([&](sycl::handler& cgh) {
          auto v_acc =
              v_sycl.template get_access<sycl::access::mode::read>(cgh);
          auto res_e_acc =
              res_e_sycl.template get_access<sycl::access::mode::read_write>(
                  cgh);
          auto res_i_acc =
              res_i_sycl.template get_access<sycl::access::mode::read_write>(
                  cgh);
          auto end_e_acc =
              end_e_sycl.template get_access<sycl::access::mode::read_write>(
                  cgh);
          auto end_i_acc =
              end_i_sycl.template get_access<sycl::access::mode::read_write>(
                  cgh);
          auto ret_type_e_acc =
              ret_type_e_sycl
                  .template get_access<sycl::access::mode::read_write>(cgh);
          auto ret_type_i_acc =
              ret_type_i_sycl
                  .template get_access<sycl::access::mode::read_write>(cgh);

          cgh.parallel_for<
              joint_scan_group_kernel<D, T, U, Group, with_init, I, OpT>>(
              executionRange, [=](sycl::nd_item<D> item) {
                Group group = get_group<Group>(item);

                T* v_begin = v_acc.get_pointer();
                T* v_end = v_begin + v_acc.size();
                U* r_e_begin = res_e_acc.get_pointer();
                U* r_i_begin = res_i_acc.get_pointer();

                auto r_e_end = joint_exclusive_scan_helper<with_init, I>(
                    group, v_begin, v_end, r_e_begin, op);
                ret_type_e_acc[0] = std::is_same_v<U*, decltype(r_e_end)>;

                auto r_i_end = joint_inclusive_scan_helper<with_init, I>(
                    group, v_begin, v_end, r_i_begin, op);
                ret_type_i_acc[0] = std::is_same_v<U*, decltype(r_i_end)>;

                end_e_acc[0] = (r_e_begin + res_e_acc.size() == r_e_end);
                end_i_acc[0] = (r_i_begin + res_i_acc.size() == r_i_end);
              });
        })
        .wait_and_throw();
  }
  CHECK(res_e_end);
  CHECK(res_i_end);
  CHECK(ret_type_e);
  CHECK(ret_type_i);
  std::vector<U> reference_e(size, U(-1));
  std::vector<U> reference_i(size, U(-1));

// FIXME: known_identity is not implemented in hipSYCL yet
#if SYCL_CTS_COMPILING_WITH_HIPSYCL
  I init_value = (with_init) ? I(init)
                 : (std::is_same_v<OpT, sycl::plus<U>>)
                     ? I{}
                     : std::numeric_limits<I>::lowest();
#else
  I init_value = (with_init) ? I(init) : sycl::known_identity<OpT, I>::value;
#endif
  std::exclusive_scan(v.begin(), v.end(), reference_e.begin(), init_value, op);
  std::inclusive_scan(v.begin(), v.end(), reference_i.begin(), op, init_value);
  for (int i = 0; i < size; i++) {
    {
      INFO("Check joint_exclusive_scan for element " + std::to_string(i));
      INFO("Result: " + std::to_string(res_e[i]));
      INFO("Expected: " + std::to_string(reference_e[i]));
      // ulp is i because there can be different rounding mods
      CHECK(is_equal_with_ulp(res_e[i], reference_e[i], i));
    }
    {
      INFO("Check joint_inclusive_scan for element " + std::to_string(i));
      INFO("Result: " + std::to_string(res_i[i]));
      INFO("Expected: " + std::to_string(reference_i[i]));
      // ulp is i because there can be different rounding mods
      CHECK(is_equal_with_ulp(res_i[i], reference_i[i], i));
    }
  }
}

/**
 * @brief Provides test for joint scans
 * @tparam D Dimension to use for group instance
 * @tparam T Type pointed by InPtr
 * @tparam U Type pointed by OutPtr
 */
template <int D, typename T, typename U>
void joint_scan_group(sycl::queue& queue) {
  INFO(" with type " + type_name<T>());
  sycl::range<D> work_group_range = util::work_group_range<D>(queue);
  size_t work_group_size = work_group_range.size();

  sycl::nd_range<D> executionRange(work_group_range, work_group_range);

  const size_t sizes[3] = {5, work_group_size / 2, 3 * work_group_size};
  for (size_t size : sizes) {
    SECTION("Check joint scan for group with sycl::plus and input size " +
            std::to_string(size)) {
      check_scan<D, T, U, sycl::group<D>, false>(queue, size, executionRange,
                                                 sycl::plus<U>());
    }

    SECTION("Check joint scan for group with sycl::maximum and input size " +
            std::to_string(size)) {
      check_scan<D, T, U, sycl::group<D>, false>(queue, size, executionRange,
                                                 sycl::maximum<U>());
    }

    SECTION("Check joint scan for sub_group with sycl::plus and input size " +
            std::to_string(size)) {
      check_scan<D, T, U, sycl::sub_group, false>(queue, size, executionRange,
                                                  sycl::plus<U>());
    }

    SECTION(
        "Check joint scan for sub_group with sycl::maximum and input size " +
        std::to_string(size)) {
      check_scan<D, T, U, sycl::sub_group, false>(queue, size, executionRange,
                                                  sycl::maximum<U>());
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
  INFO(" with type " + type_name<T>());
  sycl::range<D> work_group_range = util::work_group_range<D>(queue);
  sycl::nd_range<D> executionRange(work_group_range, work_group_range);

  size_t work_group_size = work_group_range.size();

  const size_t sizes[3] = {5, work_group_size / 2, 3 * work_group_size};
  for (size_t size : sizes) {
    SECTION("Check joint scan for group with sycl::plus and input size " +
            std::to_string(size)) {
      check_scan<D, T, U, sycl::group<D>, true, I>(queue, size, executionRange,
                                                   sycl::plus<U>());
    }

    SECTION("Check joint scan for group with sycl::maximum and input size " +
            std::to_string(size)) {
      check_scan<D, T, U, sycl::group<D>, true, I>(queue, size, executionRange,
                                                   sycl::maximum<U>());
    }

    SECTION("Check joint scan for sub_group with sycl::plus and input size " +
            std::to_string(size)) {
      check_scan<D, T, U, sycl::sub_group, true, I>(queue, size, executionRange,
                                                    sycl::plus<U>());
    }

    SECTION(
        "Check joint scan for sub_group with sycl::maximum and input size " +
        std::to_string(size)) {
      check_scan<D, T, U, sycl::sub_group, true, I>(queue, size, executionRange,
                                                    sycl::maximum<U>());
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

template <int D, typename T, typename Group, bool with_init, typename U,
          typename OpT>
class scan_over_group_kernel;

template <bool with_init, typename T, typename U, typename Group, typename OpT>
auto inclusive_scan_over_group_helper(Group group, U x, OpT op) {
  if constexpr (with_init) {
    return sycl::inclusive_scan_over_group(group, x, op, T(init));
  } else
    return sycl::inclusive_scan_over_group(group, x, op);
}

template <bool with_init, typename T, typename U, typename Group, typename OpT>
auto exclusive_scan_over_group_helper(Group group, U x, OpT op) {
  if constexpr (with_init) {
    return sycl::exclusive_scan_over_group(group, x, T(init), op);
  } else
    return sycl::exclusive_scan_over_group(group, x, op);
}

template <int D, typename T, typename Group, bool with_init, typename U = T,
          typename OpT>
void check_scan_over_group(sycl::queue& queue, sycl::range<D> range, OpT op) {
  auto range_size = range.size();
  std::vector<U> v(range_size);
  std::iota(v.begin(), v.end(), T(1));
  std::vector<T> res_e(range_size, T(-1));
  std::vector<T> res_i(range_size, T(-1));
  bool ret_type_e = false;
  bool ret_type_i = false;

  std::vector<size_t> local_id(range_size, 0);

  sycl::nd_range<D> executionRange(range, range);
  {
    sycl::buffer<U, 1> v_sycl(v.data(), sycl::range<1>(range_size));
    sycl::buffer<T, 1> res_e_sycl(res_e.data(), sycl::range<1>(range_size));
    sycl::buffer<T, 1> res_i_sycl(res_i.data(), sycl::range<1>(range_size));
    sycl::buffer<bool, 1> ret_type_e_sycl(&ret_type_e, sycl::range<1>(1));
    sycl::buffer<bool, 1> ret_type_i_sycl(&ret_type_i, sycl::range<1>(1));

    sycl::buffer<size_t, 1> local_id_sycl(local_id.data(),
                                          sycl::range<1>(range_size));

    queue
        .submit([&](sycl::handler& cgh) {
          auto v_acc =
              v_sycl.template get_access<sycl::access::mode::read>(cgh);
          auto res_e_acc =
              res_e_sycl.template get_access<sycl::access::mode::read_write>(
                  cgh);
          auto res_i_acc =
              res_i_sycl.template get_access<sycl::access::mode::read_write>(
                  cgh);
          auto ret_type_e_acc =
              ret_type_e_sycl
                  .template get_access<sycl::access::mode::read_write>(cgh);
          auto ret_type_i_acc =
              ret_type_i_sycl
                  .template get_access<sycl::access::mode::read_write>(cgh);
          auto local_id_acc =
              local_id_sycl.template get_access<sycl::access::mode::write>(cgh);

          cgh.parallel_for<
              scan_over_group_kernel<D, T, Group, with_init, U, OpT>>(
              executionRange, [=](sycl::nd_item<D> item) {
                Group group = get_group<Group>(item);

                auto index = item.get_global_linear_id();
                local_id_acc[index] = group.get_local_linear_id();

                auto res_e = exclusive_scan_over_group_helper<with_init, T>(
                    group, v_acc[index], op);
                res_e_acc[index] = res_e;
                ret_type_e_acc[0] = std::is_same_v<T, decltype(res_e)>;

                auto res_i = inclusive_scan_over_group_helper<with_init, T>(
                    group, v_acc[index], op);
                res_i_acc[index] = res_i;
                ret_type_i_acc[0] = std::is_same_v<T, decltype(res_i)>;
              });
        })
        .wait_and_throw();
  }
  CHECK(ret_type_e);
  CHECK(ret_type_i);

// FIXME: known_identity is not implemented in hipSYCL yet
#if SYCL_CTS_COMPILING_WITH_HIPSYCL
  T init_value = (with_init) ? T(init)
                 : (std::is_same_v<OpT, sycl::plus<T>>)
                     ? T{}
                     : std::numeric_limits<T>::lowest();
#else
  T init_value = (with_init) ? T(init) : sycl::known_identity<OpT, T>::value;
#endif
  for (int i = 0; i < range_size; i++) {
    int shift = i - local_id[i];
    ;
    auto startIter = v.begin() + shift;
    {
      INFO("Check exclusive_scan_over_group for element " + std::to_string(i));
      std::vector<T> reference(i + 1, T(-1));
      std::exclusive_scan(startIter, v.begin() + i + 1, reference.begin(),
                          init_value, op);
      INFO("Result: " + std::to_string(res_e[i]));
      INFO("Expected: " + std::to_string(reference[i - shift]));
      // ulp is i - shift because there can be different rounding mods
      CHECK(is_equal_with_ulp(res_e[i], reference[i - shift], i - shift));
    }
    {
      INFO("Check inclusive_scan_over_group for element " + std::to_string(i));
      std::vector<T> reference(i + 1, T(-1));
      std::inclusive_scan(startIter, v.begin() + i + 1, reference.begin(), op,
                          init_value);
      INFO("Result: " + std::to_string(res_i[i]));
      INFO("Expected: " + std::to_string(reference[i - shift]));
      // ulp is i - shift because there can be different rounding mods
      CHECK(is_equal_with_ulp(res_i[i], reference[i - shift], i - shift));
    }
  }
}

/**
 * @brief Provides test for scans over group values
 * @tparam D Dimension to use for group instance
 * @tparam T Type used for value
 */
template <int D, typename T>
void scan_over_group(sycl::queue& queue) {
  INFO(" with type " + type_name<T>());

  sycl::range<D> work_group_range = util::work_group_range<D>(queue);
  size_t work_group_size = work_group_range.size();

  SECTION("Check scan_over_group for group with sycl::plus") {
    check_scan_over_group<D, T, sycl::group<D>, false>(queue, work_group_range,
                                                       sycl::plus<T>());
  }
  SECTION("Check scan_over_group for group with sycl::maximum") {
    check_scan_over_group<D, T, sycl::group<D>, false>(queue, work_group_range,
                                                       sycl::maximum<T>());
  }

  SECTION("Check scan_over_group for sub_group with sycl::plus") {
    check_scan_over_group<D, T, sycl::sub_group, false>(queue, work_group_range,
                                                        sycl::plus<T>());
  }

  SECTION("Check scan_over_group for sub_group with sycl::maximum") {
    check_scan_over_group<D, T, sycl::sub_group, false>(queue, work_group_range,
                                                        sycl::maximum<T>());
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
  INFO(" with types " + type_name<T>() + " and " + type_name<U>());

  sycl::range<D> work_group_range = util::work_group_range<D>(queue);

  SECTION("Check scan_over_group for group with sycl::plus") {
    check_scan_over_group<D, T, sycl::group<D>, true, U>(
        queue, work_group_range, sycl::plus<T>());
  }
  SECTION("Check scan_over_group for group with sycl::maximum") {
    check_scan_over_group<D, T, sycl::group<D>, true, U>(
        queue, work_group_range, sycl::maximum<T>());
  }

  SECTION("Check scan_over_group for sub_group with sycl::plus") {
    check_scan_over_group<D, T, sycl::sub_group, true, U>(
        queue, work_group_range, sycl::plus<T>());
  }

  SECTION("Check scan_over_group for sub_group with sycl::maximum") {
    check_scan_over_group<D, T, sycl::sub_group, true, U>(
        queue, work_group_range, sycl::maximum<T>());
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

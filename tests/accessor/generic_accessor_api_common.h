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

#ifndef SYCL_CTS_GENERIC_ACCESSOR_API_COMMON_H
#define SYCL_CTS_GENERIC_ACCESSOR_API_COMMON_H
#include "accessor_common.h"

namespace generic_accessor_api_common {
using namespace sycl_cts;
using namespace accessor_tests_common;

template <typename AccT>
void test_accessor_methods(const AccT &accessor,
                           const size_t expected_byte_size,
                           const size_t expected_size,
                           const bool expected_isPlaceholder) {
  test_accessor_methods_common<AccT>(accessor, expected_byte_size,
                                     expected_size);

  {
    INFO("check is_placeholder() method");
    auto acc_isPlaceholder = accessor.is_placeholder();
    STATIC_CHECK(std::is_same_v<decltype(acc_isPlaceholder), bool>);
    CHECK(acc_isPlaceholder == expected_isPlaceholder);
  }

#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
  {
    INFO("check get_size() method");
    auto acc_get_size = accessor.get_size();
    STATIC_CHECK(std::is_same_v<decltype(acc_get_size), size_t>);
    CHECK(acc_get_size == expected_byte_size);
  }
  {
    INFO("check get_count() method");
    auto acc_get_count = accessor.get_count();
    STATIC_CHECK(std::is_same_v<decltype(acc_get_count), size_t>);
    CHECK(acc_get_count == expected_size);
  }
#endif
}

template <typename T, typename AccT>
void test_accessor_ptr_host(AccT &accessor, T expected_data) {
  {
    INFO("check get_multi_ptr() method");
    auto acc_multi_ptr_no =
        accessor.template get_multi_ptr<sycl::access::decorated::no>();
    STATIC_CHECK(
        std::is_same_v<
            decltype(acc_multi_ptr_no),
            typename AccT::template accessor_ptr<sycl::access::decorated::no>>);
    CHECK(value_operations::are_equal(*acc_multi_ptr_no, expected_data));

    auto acc_multi_ptr_yes =
        accessor.template get_multi_ptr<sycl::access::decorated::yes>();
    STATIC_CHECK(std::is_same_v<decltype(acc_multi_ptr_yes),
                                typename AccT::template accessor_ptr<
                                    sycl::access::decorated::yes>>);
    CHECK(value_operations::are_equal(*acc_multi_ptr_yes, expected_data));
  }

  {
    INFO("check get_pointer() method");
    auto acc_pointer = accessor.get_pointer();
    STATIC_CHECK(std::is_same_v<decltype(acc_pointer),
                                std::add_pointer_t<typename AccT::value_type>>);
    CHECK(value_operations::are_equal(*acc_pointer, expected_data));
  }
}

template <typename T, typename AccT, sycl::access_mode mode,
          sycl::target target>
void test_accessor_types() {
  test_accessor_types_common<T, AccT, mode>();
  if constexpr (target == sycl::target::device) {
    STATIC_CHECK(
        std::is_same_v<
            typename AccT::template accessor_ptr<sycl::access::decorated::yes>,
            sycl::multi_ptr<typename AccT::value_type,
                            sycl::access::address_space::global_space,
                            sycl::access::decorated::yes>>);
    STATIC_CHECK(
        std::is_same_v<
            typename AccT::template accessor_ptr<sycl::access::decorated::no>,
            sycl::multi_ptr<typename AccT::value_type,
                            sycl::access::address_space::global_space,
                            sycl::access::decorated::no>>);
  }
}

template <typename T, typename AccessT, typename DimensionT, typename TargetT>
class kernel_buffer_accessor;

template <typename T, typename AccessT, typename DimensionT, typename TargetT>
class kernel_offset;

template <typename T, typename AccessT, typename DimensionT, typename TargetT>
class kernel_swap;

template <typename T, typename AccessT, typename DimensionT, typename TargetT>
class run_api_tests {
  static constexpr sycl::access_mode AccessMode = AccessT::value;
  static constexpr int dims = DimensionT::value;
  static constexpr sycl::target Target = TargetT::value;
  using AccT = sycl::accessor<T, dims, AccessMode, Target>;

 public:
  void operator()(const std::string &type_name,
                  const std::string &access_mode_name,
                  const std::string &target_name) {
    auto queue = once_per_unit::get_queue();
    constexpr int buf_dims = (0 == dims) ? 1 : dims;
    auto r = util::get_cts_object::range<buf_dims>::get(1, 1, 1);

    SECTION(get_section_name<dims>(type_name, access_mode_name, target_name,
                                   "Check accessor alias types")) {
      test_accessor_types<T, AccT, AccessMode, Target>();
    }

    SECTION(get_section_name<dims>(type_name, access_mode_name, target_name,
                                   "Check api for empty accessor")) {
      queue
          .submit([&](sycl::handler &cgh) {
            AccT acc;
            test_accessor_methods(acc, 0 /* expected_byte_size*/,
                                  0 /*expected_size*/,
                                  false /*expected_isPlaceholder*/);
            if constexpr (0 < dims) {
              test_accessor_range_methods(
                  acc,
                  util::get_cts_object::range<dims>::get(0, 0,
                                                         0) /*expected_range*/,
                  sycl::id<dims>() /*&expected_offset*/);
            }
          })
          .wait_and_throw();
    }

    SECTION(
        get_section_name<dims>(type_name, access_mode_name, target_name,
                               "Check api for buffer placeholder accessor")) {
      T data = value_operations::init<T>(expected_val);
      bool res = false;
      {
        sycl::buffer<T, buf_dims> data_buf(&data, r);
        AccT acc(data_buf);

        test_accessor_methods(acc, sizeof(T) /* expected_byte_size*/,
                              1 /*expected_size*/,
                              true /*expected_isPlaceholder*/);
        if constexpr (dims > 0) {
          test_accessor_range_methods(acc,
                                      util::get_cts_object::range<dims>::get(
                                          1, 1, 1) /*expected_range*/,
                                      sycl::id<dims>() /*&expected_offset*/);
        }
      }
    }

    SECTION(get_section_name<dims>(type_name, access_mode_name, target_name,
                                   "Check api for buffer accessor")) {
      T data = value_operations::init<T>(expected_val);
      bool res = false;
      {
        sycl::buffer<T, buf_dims> data_buf(&data, r);
        sycl::buffer res_buf(&res, sycl::range(1));
        queue
            .submit([&](sycl::handler &cgh) {
              AccT acc(data_buf, cgh);

              test_accessor_methods(acc, sizeof(T) /* expected_byte_size*/,
                                    1 /*expected_size*/,
                                    false /*expected_isPlaceholder*/);
              if constexpr (0 < dims) {
                test_accessor_range_methods(
                    acc,
                    util::get_cts_object::range<dims>::get(
                        1, 1, 1) /*expected_range*/,
                    sycl::id<dims>() /*&expected_offset*/);
              }

              if constexpr (Target == sycl::target::host_task) {
                cgh.host_task([=] {
                  test_accessor_ptr_host(acc, expected_val);
                  test_begin_end_host(acc, expected_val, expected_val, false);
                  if constexpr (0 < dims) {
                    auto &acc_ref1 = acc[sycl::id<dims>()];
                    auto &acc_ref2 =
                        get_subscript_overload<T, AccT, dims>(acc, 0);
                    CHECK(value_operations::are_equal(acc_ref1, expected_val));
                    CHECK(value_operations::are_equal(acc_ref2, expected_val));
                    STATIC_CHECK(std::is_same_v<decltype(acc_ref1),
                                                typename AccT::reference>);
                    STATIC_CHECK(std::is_same_v<decltype(acc_ref2),
                                                typename AccT::reference>);
                    if constexpr (AccessMode != sycl::access_mode::read) {
                      value_operations::assign(acc_ref1, changed_val);
                      CHECK(value_operations::are_equal(acc_ref2, changed_val));
                    }
                  } else {
                    T some_data = value_operations::init<T>(expected_val);
                    typename AccT::reference dref = acc;
                    CHECK(value_operations::are_equal(some_data, dref));
                    if constexpr (AccessMode != sycl::access_mode::read) {
                      typename AccT::value_type v_data =
                          value_operations::init<typename AccT::value_type>(
                              changed_val);
                      // check method const AccT::operator=(const T& data) const
                      acc = v_data;
                      CHECK(value_operations::are_equal(dref, v_data));

                      // check method const AccT::operator=(T&& data) const
                      acc = value_operations::init<typename AccT::value_type>(
                          changed_val);
                      CHECK(value_operations::are_equal(dref, v_data));
                    }
                  }
                });
              } else {
                using kname =
                    kernel_buffer_accessor<T, AccessT, DimensionT, TargetT>;
                sycl::accessor res_acc(res_buf, cgh);
                cgh.single_task<kname>([acc, res_acc]() {
                  test_accessor_ptr_device(acc, expected_val, res_acc);
                  res_acc[0] &= test_begin_end_device(acc, expected_val,
                                                      expected_val, true);
                  if constexpr (0 < dims) {
                    auto &acc_ref1 = acc[sycl::id<dims>()];
                    auto &acc_ref2 =
                        get_subscript_overload<T, AccT, dims>(acc, 0);
                    res_acc[0] &=
                        value_operations::are_equal(acc_ref1, expected_val);
                    res_acc[0] &=
                        value_operations::are_equal(acc_ref2, expected_val);
                    res_acc[0] &= std::is_same_v<decltype(acc_ref1),
                                                 typename AccT::reference>;
                    res_acc[0] &= std::is_same_v<decltype(acc_ref2),
                                                 typename AccT::reference>;
                    if constexpr (AccessMode != sycl::access_mode::read) {
                      value_operations::assign(acc_ref1, changed_val);
                      res_acc[0] &=
                          value_operations::are_equal(acc_ref2, changed_val);
                    }
                  } else {
                    T some_data = value_operations::init<T>(expected_val);
                    typename AccT::reference dref = acc;
                    res_acc[0] &= value_operations::are_equal(some_data, dref);
                    if constexpr (AccessMode != sycl::access_mode::read) {
                      typename AccT::value_type v_data =
                          value_operations::init<typename AccT::value_type>(
                              changed_val);
                      // check method const AccT::operator=(const T& data) const
                      acc = v_data;
                      res_acc[0] &= value_operations::are_equal(dref, v_data);

                      // check method const AccT::operator=(T&& data) const
                      acc = value_operations::init<typename AccT::value_type>(
                          changed_val);
                      res_acc[0] &= value_operations::are_equal(dref, v_data);
                    }
                  }
                });
              }
            })
            .wait_and_throw();
      }
      if constexpr (Target == sycl::target::device) CHECK(res);
      if constexpr (AccessMode != sycl::access_mode::read)
        CHECK(value_operations::are_equal(data, changed_val));
    }
    if constexpr (0 < dims) {
      SECTION(
          get_section_name<dims>(type_name, access_mode_name, target_name,
                                 "Check api for ranged accessor with offset")) {
        constexpr size_t acc_range_size = 4;
        constexpr size_t buff_range_size = 8;
        constexpr size_t buff_size = (dims == 3)   ? 8 * 8 * 8
                                     : (dims == 2) ? 8 * 8
                                                   : 8;
        constexpr size_t offset = 4;
        constexpr size_t index = 2;
        int linear_index = 0;
        for (size_t i = 0; i < dims; i++) {
          linear_index += (offset + index) * pow(buff_range_size, dims - i - 1);
        }
        auto acc_range = util::get_cts_object::range<dims>::get(
            acc_range_size, acc_range_size, acc_range_size);
        auto buff_range = util::get_cts_object::range<dims>::get(
            buff_range_size, buff_range_size, buff_range_size);
        auto offset_id =
            util::get_cts_object::id<dims>::get(offset, offset, offset);
        std::remove_const_t<T> data[buff_size];
        std::generate(data, (data + buff_range.size()), [i = 0]() mutable {
          return value_operations::init<T>(i++);
        });
        bool res = false;
        {
          sycl::buffer<T, dims> data_buf(data, buff_range);
          sycl::buffer res_buf(&res, sycl::range(1));
          queue
              .submit([&](sycl::handler &cgh) {
                AccT acc(data_buf, cgh, acc_range, offset_id);
                test_accessor_methods(
                    acc, sizeof(T) * acc_range.size() /* expected_byte_size*/,
                    acc_range.size() /*expected_size*/,
                    false /*expected_isPlaceholder*/);
                test_accessor_range_methods(acc, acc_range /*expected_range*/,
                                            offset_id /*&expected_offset*/);

                if constexpr (Target == sycl::target::host_task) {
                  cgh.host_task([=] {
                    test_accessor_ptr_host(acc, T());
                    test_begin_end_host(
                        acc, value_operations::init<T>(0),
                        value_operations::init<T>(buff_size - 1), false);
                    auto &acc_ref1 =
                        get_subscript_overload<T, AccT, dims>(acc, index);
                    auto &acc_ref2 = acc[sycl::id<dims>()];
                    CHECK(value_operations::are_equal(acc_ref1, linear_index));
                    CHECK(value_operations::are_equal(acc_ref2, 0));
                    if constexpr (AccessMode != sycl::access_mode::read) {
                      value_operations::assign(acc_ref1, changed_val);
                      value_operations::assign(acc_ref2, expected_val);
                    }
                  });
                } else {
                  using kname = kernel_offset<T, AccessT, DimensionT, TargetT>;
                  sycl::accessor res_acc(res_buf, cgh);
                  cgh.single_task<kname>([=]() {
                    test_accessor_ptr_device(acc, T(), res_acc);
                    res_acc[0] &= test_begin_end_device(
                        acc, value_operations::init<T>(0),
                        value_operations::init<T>(buff_size - 1), true);
                    auto &acc_ref1 =
                        get_subscript_overload<T, AccT, dims>(acc, index);
                    auto &acc_ref2 = acc[sycl::id<dims>()];
                    res_acc[0] &=
                        value_operations::are_equal(acc_ref1, linear_index);
                    res_acc[0] &= value_operations::are_equal(acc_ref2, 0);
                    if constexpr (AccessMode != sycl::access_mode::read) {
                      value_operations::assign(acc_ref1, changed_val);
                      value_operations::assign(acc_ref2, expected_val);
                    }
                  });
                }
              })
              .wait_and_throw();
        }
        if constexpr (Target == sycl::target::device) CHECK(res);
        if constexpr (AccessMode != sycl::access_mode::read) {
          CHECK(value_operations::are_equal(data[linear_index], changed_val));
          CHECK(value_operations::are_equal(data[0], expected_val));
        }
      }
    }
    SECTION(get_section_name<dims>(type_name, access_mode_name, target_name,
                                   "Check swap for accessor")) {
      T data1 = value_operations::init<T>(expected_val);
      T data2 = value_operations::init<T>(changed_val);
      bool res = false;
      {
        sycl::buffer res_buf(&res, sycl::range(1));
        sycl::buffer<T, buf_dims> data_buf1(&data1, r);
        sycl::buffer<T, buf_dims> data_buf2(&data2, r);
        queue
            .submit([&](sycl::handler &cgh) {
              AccT acc1(data_buf1, cgh);
              AccT acc2(data_buf2, cgh);
              acc1.swap(acc2);
              if constexpr (Target == sycl::target::host_task) {
                cgh.host_task([=] {
                  typename AccT::reference acc_ref1 =
                      get_accessor_reference<dims>(acc1);
                  typename AccT::reference acc_ref2 =
                      get_accessor_reference<dims>(acc2);
                  CHECK(value_operations::are_equal(acc_ref1, changed_val));
                  CHECK(value_operations::are_equal(acc_ref2, expected_val));
                  if constexpr (AccessMode != sycl::access_mode::read) {
                    value_operations::assign(acc_ref1, expected_val);
                    value_operations::assign(acc_ref2, changed_val);
                  }
                });
              } else {
                using kname = kernel_swap<T, AccessT, DimensionT, TargetT>;
                sycl::accessor res_acc(res_buf, cgh);
                cgh.single_task<kname>([=]() {
                  if constexpr (0 < dims) {
                    typename AccT::reference acc_ref1 =
                        get_accessor_reference<dims>(acc1);
                    typename AccT::reference acc_ref2 =
                        get_accessor_reference<dims>(acc2);
                    res_acc[0] =
                        value_operations::are_equal(acc_ref1, changed_val);
                    res_acc[0] &=
                        value_operations::are_equal(acc_ref2, expected_val);
                    if constexpr (AccessMode != sycl::access_mode::read) {
                      value_operations::assign(acc_ref1, expected_val);
                      value_operations::assign(acc_ref2, changed_val);
                    }
                  }
                });
              }
            })
            .wait_and_throw();
      }
      if constexpr (Target == sycl::target::device) CHECK(res);
      if constexpr (AccessMode != sycl::access_mode::read) {
        CHECK(value_operations::are_equal(data1, changed_val));
        CHECK(value_operations::are_equal(data2, expected_val));
      } else {
        CHECK(value_operations::are_equal(data1, expected_val));
        CHECK(value_operations::are_equal(data2, changed_val));
      }
    }
  }
};

template <typename T>
class run_generic_api_for_type {
 public:
  void operator()(const std::string &type_name) {
    const auto access_modes = get_access_modes();
    const auto dimensions = get_all_dimensions();
    const auto targets = get_targets();

    // To handle cases when class was called from functions
    // like for_all_types_vectors_marray or for_all_device_copyable_std_containers.
    // This will wrap string with type T to string with container<T> if T is
    // an array or other kind of container.
    auto actual_type_name = type_name_string<T>::get(type_name);

    for_all_combinations<run_api_tests, T>(access_modes, dimensions, targets,
                                           actual_type_name);

    // For covering const types
    actual_type_name = std::string("const ") + actual_type_name;
    // const T can be only with access_mode::read
    const auto read_only_acc_mode =
        value_pack<sycl::access_mode, sycl::access_mode::read>::generate_named(
            "access_mode::read");
    for_all_combinations<run_api_tests, const T>(read_only_acc_mode, dimensions,
                                                 targets, actual_type_name);
  }
};
}  // namespace generic_accessor_api_common
#endif  // SYCL_CTS_GENERIC_ACCESSOR_API_COMMON_H

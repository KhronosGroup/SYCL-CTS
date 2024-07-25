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
//  Provides tests for sycl::host_accessor constructors
//
*******************************************************************************/
#ifndef SYCL_CTS_HOST_ACCESSOR_CONSTRUCTORS_H
#define SYCL_CTS_HOST_ACCESSOR_CONSTRUCTORS_H
#include "catch2/catch_test_macros.hpp"

#include "../accessor_basic/accessor_common.h"

namespace host_accessor_constructors {
using namespace sycl_cts;
using namespace accessor_tests_common;

constexpr accessor_type AccType = accessor_type::host_accessor;

template <typename DataT, int Dimension, sycl::access_mode AccessMode>
void test_default_constructor(const std::string& access_mode_name,
                              const std::string& type_name) {
  const auto section_name = get_section_name<Dimension>(
      type_name, access_mode_name, "Default constructor");

  SECTION(section_name) {
    auto get_acc_functor = [] {
      return sycl::host_accessor<DataT, Dimension, AccessMode>();
    };
    check_def_constructor<AccType, DataT, Dimension, AccessMode>(
        get_acc_functor);
  }
}

template <typename DataT, sycl::access_mode AccessMode>
void test_zero_dimension_buffer_constructor(const std::string& access_mode_name,
                                            const std::string& type_name) {
  const auto section_name = get_section_name<0>(type_name, access_mode_name,
                                                "Zero dimension constructor");

  SECTION(section_name) {
    auto get_acc_functor = [](sycl::buffer<DataT, 1> data_buf) {
      return sycl::host_accessor<DataT, 0, AccessMode>(data_buf);
    };
    check_zero_dim_constructor<AccType, DataT, AccessMode>(get_acc_functor);
  }
}

template <typename DataT, int Dimension, sycl::access_mode AccessMode>
void test_common_buffer_constructors(const std::string& access_mode_name,
                                     const std::string& type_name) {
  constexpr int buf_dims = (0 == Dimension) ? 1 : Dimension;
  auto r = util::get_cts_object::range<buf_dims>::get(1, 1, 1);
  const auto offset = sycl::id<buf_dims>();
  const auto r_zero = util::get_cts_object::range<buf_dims>::get(0, 0, 0);

  auto section_name = get_section_name<Dimension>(type_name, access_mode_name,
                                                  "From buffer constructor");

  SECTION(section_name) {
    auto get_acc_functor = [](sycl::buffer<DataT, buf_dims> data_buf) {
      return sycl::host_accessor<DataT, Dimension, AccessMode>(data_buf);
    };
    check_common_constructor<AccType, DataT, Dimension, AccessMode>(
        get_acc_functor);
  }

  section_name = get_section_name<Dimension>(
      type_name, access_mode_name, "From zero-length buffer constructor");

  SECTION(section_name) {
    auto get_acc_functor = [](sycl::buffer<DataT, buf_dims>& data_buf) {
      return sycl::host_accessor<DataT, Dimension, AccessMode>(data_buf);
    };
    check_zero_length_buffer_constructor<AccType, DataT, Dimension, AccessMode>(
        get_acc_functor);
  }

  if constexpr (0 != Dimension) {
    section_name = get_section_name<Dimension>(
        type_name, access_mode_name, "From buffer and range constructor");

    SECTION(section_name) {
      auto get_acc_functor = [r](sycl::buffer<DataT, buf_dims> data_buf) {
        return sycl::host_accessor<DataT, Dimension, AccessMode>(data_buf, r);
      };
      check_common_constructor<AccType, DataT, Dimension, AccessMode>(
          get_acc_functor);
    }

    section_name = get_section_name<Dimension>(
        type_name, access_mode_name,
        "From zero-length buffer and range constructor");

    SECTION(section_name) {
      auto get_acc_functor = [r_zero](sycl::buffer<DataT, buf_dims> data_buf) {
        return sycl::host_accessor<DataT, Dimension, AccessMode>(data_buf,
                                                                 r_zero);
      };
      check_zero_length_buffer_constructor<AccType, DataT, Dimension,
                                           AccessMode>(get_acc_functor);
    }

    section_name =
        get_section_name<Dimension>(type_name, access_mode_name,
                                    "From buffer,range and offset constructor");

    SECTION(section_name) {
      auto get_acc_functor = [r,
                              offset](sycl::buffer<DataT, buf_dims> data_buf) {
        return sycl::host_accessor<DataT, Dimension, AccessMode>(data_buf, r,
                                                                 offset);
      };
      check_common_constructor<AccType, DataT, Dimension, AccessMode>(
          get_acc_functor);
    }

    section_name = get_section_name<Dimension>(
        type_name, access_mode_name,
        "From zero-length buffer,range and offset constructor");

    SECTION(section_name) {
      auto get_acc_functor = [r_zero,
                              offset](sycl::buffer<DataT, buf_dims> data_buf) {
        return sycl::host_accessor<DataT, Dimension, AccessMode>(
            data_buf, r_zero, offset);
      };
      check_zero_length_buffer_constructor<AccType, DataT, Dimension,
                                           AccessMode>(get_acc_functor);
    }
  }
}

template <typename DataT, int Dimension, sycl::access_mode AccessMode>
void test_common_buffer_constructors_tag_t_deduction(
    const std::string& access_mode_name, const std::string& type_name) {
  const auto r = util::get_cts_object::range<Dimension>::get(1, 1, 1);
  const auto offset = sycl::id<Dimension>();
  const auto r_zero = util::get_cts_object::range<Dimension>::get(0, 0, 0);
  auto tagT = tag_factory<AccType>::get_tag<AccessMode>();

  auto section_name = get_section_name<Dimension>(
      type_name, access_mode_name, "TagT deduction from buffer constructor");

  SECTION(section_name) {
    auto get_acc_functor = [tagT](sycl::buffer<DataT, Dimension> data_buf) {
      return sycl::host_accessor<DataT, Dimension, AccessMode>(data_buf, tagT);
    };
    check_common_constructor<AccType, DataT, Dimension, AccessMode>(
        get_acc_functor);
  }

  section_name = get_section_name<Dimension>(
      type_name, access_mode_name,
      "TagT deduction from zero-length buffer constructor");

  SECTION(section_name) {
    auto get_acc_functor = [tagT](sycl::buffer<DataT, Dimension> data_buf) {
      return sycl::host_accessor<DataT, Dimension, AccessMode>(data_buf, tagT);
    };
    check_zero_length_buffer_constructor<AccType, DataT, Dimension, AccessMode>(
        get_acc_functor);
  }

  section_name = get_section_name<Dimension>(
      type_name, access_mode_name,
      "TagT deduction from buffer and range constructor");

  SECTION(section_name) {
    auto get_acc_functor = [r, tagT](sycl::buffer<DataT, Dimension> data_buf) {
      return sycl::host_accessor(data_buf, r, tagT);
    };
    check_common_constructor<AccType, DataT, Dimension, AccessMode>(
        get_acc_functor);
  }

  section_name = get_section_name<Dimension>(
      type_name, access_mode_name,
      "TagT deduction from zero-length buffer and range constructor");

  SECTION(section_name) {
    auto get_acc_functor = [r_zero,
                            tagT](sycl::buffer<DataT, Dimension> data_buf) {
      return sycl::host_accessor(data_buf, r_zero, tagT);
    };
    check_zero_length_buffer_constructor<AccType, DataT, Dimension, AccessMode>(
        get_acc_functor);
  }

  section_name =
      get_section_name<Dimension>(type_name, access_mode_name,
                                  "TagT deduction from buffer,range and "
                                  "offset constructor");

  SECTION(section_name) {
    auto get_acc_functor = [r, offset,
                            tagT](sycl::buffer<DataT, Dimension> data_buf) {
      return sycl::host_accessor(data_buf, r, offset, tagT);
    };
    check_common_constructor<AccType, DataT, Dimension, AccessMode>(
        get_acc_functor);
  }

  section_name = get_section_name<Dimension>(
      type_name, access_mode_name,
      "TagT deduction from zero-length buffer,range and "
      "offset constructor");

  SECTION(section_name) {
    auto get_acc_functor = [r_zero, offset,
                            tagT](sycl::buffer<DataT, Dimension> data_buf) {
      return sycl::host_accessor(data_buf, r_zero, offset, tagT);
    };
    check_zero_length_buffer_constructor<AccType, DataT, Dimension, AccessMode>(
        get_acc_functor);
  }
}

template <typename T, typename AccessModeT, typename DimensionT>
class run_tests_constructors {
  static constexpr sycl::access_mode AccessMode = AccessModeT::value;
  static constexpr int Dimension = DimensionT::value;

 public:
  void operator()(const std::string& type_name,
                  const std::string& access_mode_name) {
    if constexpr (0 == Dimension) {
      test_zero_dimension_buffer_constructor<T, AccessMode>(access_mode_name,
                                                            type_name);
    }
    test_default_constructor<T, Dimension, AccessMode>(access_mode_name,
                                                       type_name);
    test_common_buffer_constructors<T, Dimension, AccessMode>(access_mode_name,
                                                              type_name);
    if constexpr (0 != Dimension) {
      test_common_buffer_constructors_tag_t_deduction<T, Dimension, AccessMode>(
          access_mode_name, type_name);
    }
  }
};

using test_combinations =
    typename get_combinations<access_modes_pack, all_dimensions_pack>::type;

template <typename T, typename ArgCombination>
class run_host_constructors_test {
 public:
  void operator()(const std::string& type_name) {
    // Get the packs from the test combination type.
    using AccessModePack = std::tuple_element_t<0, ArgCombination>;
    using DimensionsPack = std::tuple_element_t<1, ArgCombination>;

    // Type packs instances have to be const, otherwise for_all_combination
    // will not compile
    const auto access_modes = AccessModePack::generate_named();
    const auto dimensions = DimensionsPack::generate_unnamed();

    // To handle cases when class was called from functions
    // like for_all_types_vectors_marray or for_all_device_copyable_std_containers.
    // This will wrap string with type T to string with container<T> if T is
    // an array or other kind of container.
    auto actual_type_name = type_name_string<T>::get(type_name);

    for_all_combinations<run_tests_constructors, T>(access_modes, dimensions,
                                                    actual_type_name);

    // For covering const types
    actual_type_name = std::string("const ") + actual_type_name;
    // const T can be only with access_mode::read
    const auto read_only_acc_mode =
        value_pack<sycl::access_mode, sycl::access_mode::read>::generate_named(
            "access_mode::read");
    for_all_combinations<run_tests_constructors, const T>(
        read_only_acc_mode, dimensions, actual_type_name);
  }
};
}  // namespace host_accessor_constructors

#endif  // SYCL_CTS_HOST_ACCESSOR_CONSTRUCTORS_H

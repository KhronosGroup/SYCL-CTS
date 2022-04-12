/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for generic sycl::accessor constructors
//
*******************************************************************************/
#ifndef SYCL_CTS_GENERIC_ACCESSOR_CONSTRUCTORS_TARGET_DEVICE_H
#define SYCL_CTS_GENERIC_ACCESSOR_CONSTRUCTORS_TARGET_DEVICE_H
#include "accessor_common.h"

#include "catch2/catch_test_macros.hpp"

namespace generic_accessor_constructors {
using namespace sycl_cts;
using namespace accessor_tests_common;

constexpr accessor_type AccTypeT = accessor_type::generic_accessor;

template <typename DataT, int DimensionT, sycl::access_mode AccessModeT,
          sycl::target TargetT>
void test_default_constructor(const std::string& type_name,
                              const std::string& access_mode_name,
                              const std::string& target_name) {
  const auto section_name = get_section_name<DimensionT>(
      type_name, access_mode_name, target_name, "Default constructor");

  SECTION(section_name) {
    auto get_acc_functor = [] {
      return sycl::accessor<DataT, DimensionT, AccessModeT, TargetT>();
    };
    check_def_constructor<AccTypeT, DataT, DimensionT, AccessModeT, TargetT>(
        get_acc_functor);
  }
}

template <typename DataT, sycl::access_mode AccessModeT, sycl::target TargetT>
void test_zero_dimension_buffer_constructor(const std::string& type_name,
                                            const std::string& access_mode_name,
                                            const std::string& target_name) {
  const auto section_name = get_section_name<0>(
      type_name, access_mode_name, target_name, "Zero dimension constructor");

  SECTION(section_name) {
    auto get_acc_functor = [](sycl::buffer<DataT, 1>& data_buf,
                              sycl::handler& cgh) {
      return sycl::accessor<DataT, 0, AccessModeT, TargetT>(data_buf, cgh);
    };
    check_zero_dim_constructor<AccTypeT, DataT, AccessModeT, TargetT>(
        get_acc_functor);
  }
}

template <typename DataT, int DimensionT, sycl::access_mode AccessModeT,
          sycl::target TargetT>
void test_common_buffer_constructors(const std::string& type_name,
                                     const std::string& access_mode_name,
                                     const std::string& target_name) {
  auto r = util::get_cts_object::range<DimensionT>::get(1, 1, 1);
  auto offset = util::get_cts_object::id<DimensionT>::get(0, 0, 0);

  auto section_name = get_section_name<DimensionT>(
      type_name, access_mode_name, target_name, "From buffer constructor");

  SECTION(section_name) {
    auto get_acc_functor = [](sycl::buffer<DataT, DimensionT>& data_buf,
                              sycl::handler& cgh) {
      return sycl::accessor<DataT, DimensionT, AccessModeT, TargetT>(data_buf,
                                                                     cgh);
    };
    check_common_constructor<AccTypeT, DataT, DimensionT, AccessModeT, TargetT>(
        get_acc_functor);
  }

  section_name =
      get_section_name<DimensionT>(type_name, access_mode_name, target_name,
                                   "From buffer and range constructor");

  SECTION(section_name) {
    auto get_acc_functor = [r](sycl::buffer<DataT, DimensionT>& data_buf,
                               sycl::handler& cgh) {
      return sycl::accessor<DataT, DimensionT, AccessModeT, TargetT>(data_buf,
                                                                     cgh, r);
    };
    check_common_constructor<AccTypeT, DataT, DimensionT, AccessModeT, TargetT>(
        get_acc_functor);
  }

  section_name =
      get_section_name<DimensionT>(type_name, access_mode_name, target_name,
                                   "From buffer, range and offset constructor");

  SECTION(section_name) {
    auto get_acc_functor = [r, offset](
                               sycl::buffer<DataT, DimensionT>& data_buf,
                               sycl::handler& cgh) {
      return sycl::accessor<DataT, DimensionT, AccessModeT, TargetT>(
          data_buf, cgh, r, offset);
    };
    check_common_constructor<AccTypeT, DataT, DimensionT, AccessModeT, TargetT>(
        get_acc_functor);
  }
}

template <typename DataT, int DimensionT, sycl::access_mode AccessModeT,
          sycl::target TargetT>
void test_common_buffer_constructors_tag_t_deduction(
    const std::string& type_name, const std::string& access_mode_name,
    const std::string& target_name) {
  auto r = util::get_cts_object::range<DimensionT>::get(1, 1, 1);
  auto offset = util::get_cts_object::id<DimensionT>::get(0, 0, 0);
  const auto tag = get_tag<AccessModeT, TargetT>();

  auto section_name =
      get_section_name<DimensionT>(type_name, access_mode_name, target_name,
                                   "TagT deduction from buffer constructor");

  SECTION(section_name) {
    auto get_acc_functor = [tag](sycl::buffer<DataT, DimensionT>& data_buf,
                                 sycl::handler& cgh) {
      return sycl::accessor(data_buf, cgh, tag);
    };
    check_common_constructor<AccTypeT, DataT, DimensionT, AccessModeT, TargetT>(
        get_acc_functor);
  }

  section_name = get_section_name<DimensionT>(
      type_name, access_mode_name, target_name,
      "TagT deduction from buffer and range constructor");

  SECTION(section_name) {
    auto get_acc_functor = [r, tag](sycl::buffer<DataT, DimensionT>& data_buf,
                                    sycl::handler& cgh) {
      return sycl::accessor(data_buf, cgh, r, tag);
    };
    check_common_constructor<AccTypeT, DataT, DimensionT, AccessModeT, TargetT>(
        get_acc_functor);
  }

  section_name =
      get_section_name<DimensionT>(type_name, access_mode_name, target_name,
                                   "TagT deduction from buffer, range and "
                                   "offset constructor");

  SECTION(section_name) {
    auto get_acc_functor = [r, offset, tag](
                               sycl::buffer<DataT, DimensionT>& data_buf,
                               sycl::handler& cgh) {
      return sycl::accessor(data_buf, cgh, r, offset, tag);
    };
    check_common_constructor<AccTypeT, DataT, DimensionT, AccessModeT, TargetT>(
        get_acc_functor);
  }
}

template <typename DataT, int DimensionT, sycl::access_mode AccessModeT,
          sycl::target TargetT>
void test_placeholder_constructors(const std::string& type_name,
                                   const std::string& access_mode_name,
                                   const std::string& target_name) {
  auto r = util::get_cts_object::range<DimensionT>::get(1, 1, 1);
  auto offset = util::get_cts_object::id<DimensionT>::get(0, 0, 0);

  auto section_name =
      get_section_name<DimensionT>(type_name, access_mode_name, target_name,
                                   "From buffer placeholder constructor");

  SECTION(section_name) {
    auto get_acc_functor = [](sycl::buffer<DataT, DimensionT>& data_buf,
                              sycl::handler& cgh) {
      return sycl::accessor<DataT, DimensionT, AccessModeT, TargetT>(data_buf);
    };
    check_common_constructor<AccTypeT, DataT, DimensionT, AccessModeT, TargetT>(
        get_acc_functor);
  }

  section_name = get_section_name<DimensionT>(
      type_name, access_mode_name, target_name,
      "From buffer and range placeholder constructor");

  SECTION(section_name) {
    auto get_acc_functor = [r](sycl::buffer<DataT, DimensionT>& data_buf,
                               sycl::handler& cgh) {
      return sycl::accessor<DataT, DimensionT, AccessModeT, TargetT>(data_buf,
                                                                     r);
    };
    check_common_constructor<AccTypeT, DataT, DimensionT, AccessModeT, TargetT>(
        get_acc_functor);
  }

  section_name = get_section_name<DimensionT>(
      type_name, access_mode_name, target_name,
      "From buffer, range and offset placeholder constructor");

  SECTION(section_name) {
    auto get_acc_functor = [r, offset](
                               sycl::buffer<DataT, DimensionT>& data_buf,
                               sycl::handler& cgh) {
      return sycl::accessor<DataT, DimensionT, AccessModeT, TargetT>(data_buf,
                                                                     r, offset);
    };
    check_common_constructor<AccTypeT, DataT, DimensionT, AccessModeT, TargetT>(
        get_acc_functor);
  }
}

template <typename DataT, int DimensionT, sycl::access_mode AccessModeT,
          sycl::target TargetT>
void test_placeholder_accessors_exception(const std::string& type_name,
                                          const std::string& access_mode_name,
                                          const std::string& target_name) {
  auto r = util::get_cts_object::range<DimensionT>::get(1, 1, 1);
  auto offset = util::get_cts_object::id<DimensionT>::get(0, 0, 0);

  auto section_name =
      get_section_name<DimensionT>(type_name, access_mode_name, target_name,
                                   "From buffer placeholder constructor");

  SECTION(section_name) {
    auto get_acc_functor = [](sycl::buffer<DataT, DimensionT>& data_buf) {
      return sycl::accessor<DataT, DimensionT, AccessModeT, TargetT>(data_buf);
    };
    check_placeholder_accessor_exception<AccTypeT, DataT, DimensionT,
                                         AccessModeT, TargetT>(get_acc_functor);
  }

  section_name = get_section_name<DimensionT>(
      type_name, access_mode_name, target_name,
      "From buffer and range placeholder constructor");

  SECTION(section_name) {
    auto get_acc_functor = [r](sycl::buffer<DataT, DimensionT>& data_buf) {
      return sycl::accessor<DataT, DimensionT, AccessModeT, TargetT>(data_buf,
                                                                     r);
    };
    check_placeholder_accessor_exception<AccTypeT, DataT, DimensionT,
                                         AccessModeT, TargetT>(get_acc_functor);
  }

  section_name = get_section_name<DimensionT>(
      type_name, access_mode_name, target_name,
      "From buffer, range and offset placeholder constructor");

  SECTION(section_name) {
    auto get_acc_functor = [r,
                            offset](sycl::buffer<DataT, DimensionT>& data_buf) {
      return sycl::accessor<DataT, DimensionT, AccessModeT, TargetT>(data_buf,
                                                                     r, offset);
    };
    check_placeholder_accessor_exception<AccTypeT, DataT, DimensionT,
                                         AccessModeT, TargetT>(get_acc_functor);
  }
}

template <typename T, typename AccessTypeT, typename TargetTypeT,
          typename DimensionTypeT>
class run_tests_constructors {
  static constexpr sycl::access_mode AccessModeT = AccessTypeT::value;
  static constexpr int DimensionT = DimensionTypeT::value;
  static constexpr sycl::target TargetT = TargetTypeT::value;

 public:
  void operator()(const std::string& type_name,
                  const std::string& access_mode_name,
                  const std::string& target_name) {
    test_zero_dimension_buffer_constructor<T, AccessModeT, TargetT>(
        type_name, access_mode_name, target_name);
    test_default_constructor<T, DimensionT, AccessModeT, TargetT>(
        type_name, access_mode_name, target_name);
    test_common_buffer_constructors<T, DimensionT, AccessModeT, TargetT>(
        type_name, access_mode_name, target_name);
    test_common_buffer_constructors_tag_t_deduction<T, DimensionT, AccessModeT,
                                                    TargetT>(
        type_name, access_mode_name, target_name);
    test_placeholder_constructors<T, DimensionT, AccessModeT, TargetT>(
        type_name, access_mode_name, target_name);
    test_placeholder_accessors_exception<T, DimensionT, AccessModeT, TargetT>(
        type_name, access_mode_name, target_name);
  }
};

template <typename T>
class run_generic_constructors_test {
 public:
  void operator()(const std::string& type_name) {
    // Type packs instances have to be const, otherwise for_all_combination will
    // not compile
    const auto access_modes = get_access_modes();
    const auto dimensions = get_dimensions();
    const auto targets = get_targets();
    const auto cur_type =
        named_type_pack<T>::generate(type_name_string<T>::get(type_name));

    for_all_combinations<run_tests_constructors>(cur_type, access_modes,
                                                 targets, dimensions);
                                                 
    // For covering const types
    const auto const_cur_type = named_type_pack<const T>::generate(
        "const " + type_name_string<T>::get(type_name));
    // const T can be only with access_mode::read
    const auto read_only_acc_mode =
        value_pack<sycl::access_mode, sycl::access_mode::read>::generate_named(
            "access_mode::read");
    for_all_combinations<run_tests_constructors>(
        const_cur_type, read_only_acc_mode, targets, dimensions);
  }
};
}  // namespace generic_accessor_constructors
#endif  // SYCL_CTS_GENERIC_ACCESSOR_CONSTRUCTORS_TARGET_DEVICE_H

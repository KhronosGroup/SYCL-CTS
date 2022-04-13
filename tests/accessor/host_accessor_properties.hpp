/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for the properties of sycl::host_accessor
//
*******************************************************************************/
#ifndef SYCL_CTS_HOST_ACCESSOR_PROPERTIES_H
#define SYCL_CTS_HOST_ACCESSOR_PROPERTIES_H
#include "accessor_common.h"

namespace host_accessor_properties {
using namespace sycl_cts;
using namespace accessor_tests_common;

constexpr accessor_type AccTypeT = accessor_type::host_accessor;

template <typename DataT, int DimensionT, sycl::access_mode AccessModeT>
void test_constructor_with_no_init(const std::string& type_name,
                                   const std::string& access_mode_name) {
  auto r = util::get_cts_object::range<DimensionT>::get(1, 1, 1);
  auto offset = sycl::id<DimensionT>();
  const sycl::property_list prop_list(sycl::no_init);

  auto section_name = get_section_name<DimensionT>(
      type_name, access_mode_name,
      "From buffer constructor with no_init property");

  SECTION(section_name) {
    auto construct_acc =
        [&prop_list](sycl::buffer<DataT, DimensionT> data_buf) {
          return sycl_stub::host_accessor<DataT, DimensionT, AccessModeT>(
              data_buf, prop_list);
        };

    check_no_init_prop<AccTypeT, DataT, DimensionT, AccessModeT>(construct_acc);
  }

  section_name = get_section_name<DimensionT>(
      type_name, access_mode_name,
      "From buffer and range constructor with no_init property");

  SECTION(section_name) {
    auto construct_acc = [&prop_list,
                          r](sycl::buffer<DataT, DimensionT> data_buf) {
      return sycl_stub::host_accessor<DataT, DimensionT, AccessModeT>(
          data_buf, r, prop_list);
    };

    check_no_init_prop<AccTypeT, DataT, DimensionT, AccessModeT>(construct_acc);
  }

  section_name = get_section_name<DimensionT>(
      type_name, access_mode_name,
      "From buffer,range and offset constructor with no_init property");

  SECTION(section_name) {
    auto construct_acc = [&prop_list, r,
                          offset](sycl::buffer<DataT, DimensionT> data_buf) {
      return sycl_stub::host_accessor<DataT, DimensionT, AccessModeT>(
          data_buf, r, offset, prop_list);
    };

    check_no_init_prop<AccTypeT, DataT, DimensionT, AccessModeT>(construct_acc);
  }
}

template <typename DataT, int DimensionT>
void test_exception(const std::string& type_name) {
  auto r = util::get_cts_object::range<DimensionT>::get(1, 1, 1);
  auto offset = sycl::id<DimensionT>();
  const sycl::property_list prop_list(sycl::no_init);

  auto section_name = get_section_name<DimensionT>(
      type_name, "access_mode::read",
      "Expecting exception when attempting to construct accessor from buffer "
      "with no_init property and access_mode::read");
  SECTION(section_name) {
    auto construct_acc =
        [&prop_list](sycl::buffer<DataT, DimensionT> data_buf) {
          sycl_stub::host_accessor<DataT, DimensionT, sycl::access_mode::read>(
              data_buf, prop_list);
        };
    check_no_init_prop_exception<AccTypeT, DataT, DimensionT>(construct_acc);
  }

  section_name = get_section_name<DimensionT>(
      type_name, "access_mode::read",
      "Expecting exception when attempting to construct accessor from buffer "
      "and range with no_init property and access_mode::read");
  SECTION(section_name) {
    auto construct_acc = [&prop_list,
                          r](sycl::buffer<DataT, DimensionT> data_buf) {
      sycl_stub::host_accessor<DataT, DimensionT, sycl::access_mode::read>(
          data_buf, r, prop_list);
    };
    check_no_init_prop_exception<AccTypeT, DataT, DimensionT>(construct_acc);
  }

  section_name = get_section_name<DimensionT>(
      type_name, "access_mode::read",
      "Expecting exception when attempting to construct accessor from buffer, "
      "range and offset with no_init property and access_mode::read");
  SECTION(section_name) {
    auto construct_acc = [&prop_list, r,
                          offset](sycl::buffer<DataT, DimensionT> data_buf) {
      sycl_stub::host_accessor<DataT, DimensionT, sycl::access_mode::read>(
          data_buf, r, offset, prop_list);
    };
    check_no_init_prop_exception<AccTypeT, DataT, DimensionT>(construct_acc);
  }
}

template <typename DataT, int DimensionT, sycl::access_mode AccessModeT>
void test_property_member_functions(const std::string& type_name,
                                    const std::string& access_mode_name) {
  auto r = util::get_cts_object::range<DimensionT>::get(1, 1, 1);
  auto offset = sycl::id<DimensionT>();
  const sycl::property_list prop_list(sycl::no_init);

  const auto construct_acc =
      [&prop_list](sycl::buffer<DataT, DimensionT> data_buf) {
        return sycl_stub::host_accessor<DataT, DimensionT, AccessModeT>(
            data_buf, prop_list);
      };

  auto section_name = get_section_name<DimensionT>(
      type_name, access_mode_name, "has_property member function invocation");
  SECTION(section_name) {
    check_has_property_member_func<DataT, DimensionT, sycl::property::no_init>(
        construct_acc);
  }
  section_name = get_section_name<DimensionT>(
      type_name, access_mode_name, "get_property member function invocation");
  SECTION(section_name) {
    check_get_property_member_func<DataT, DimensionT, sycl::property::no_init>(
        construct_acc);
  }
}

template <typename T, typename AccessTypeT, typename DimensionTypeT>
class run_tests_properties {
  static constexpr sycl::access_mode AccessModeT = AccessTypeT::value;
  static constexpr int DimensionT = DimensionTypeT::value;

 public:
  void operator()(const std::string& type_name,
                  const std::string& access_mode_name) {
    test_constructor_with_no_init<T, DimensionT, AccessModeT>(type_name,
                                                              access_mode_name);

    test_property_member_functions<T, DimensionT, AccessModeT>(
        type_name, access_mode_name);

    // In order not to run again with same parameters
    if constexpr (AccessModeT == sycl::access_mode::read) {
      test_exception<T, DimensionT>(type_name);
    }
  }
};

template <typename T>
class run_host_properties_tests {
 public:
  void operator()(const std::string& type_name) {
    // Type packs instances have to be const, otherwise for_all_combination will
    // not compile
    const auto access_modes = get_access_modes();
    const auto dimensions = get_dimensions();
    const auto cur_type =
        named_type_pack<T>::generate(type_name_string<T>::get(type_name));

    for_all_combinations<run_tests_properties>(cur_type, access_modes,
                                               dimensions);
  }
};
}  // namespace host_accessor_properties

#endif  // SYCL_CTS_HOST_ACCESSOR_PROPERTIES_H

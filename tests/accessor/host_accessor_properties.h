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

constexpr accessor_type AccType = accessor_type::host_accessor;

template <typename DataT, int Dimension, sycl::access_mode AccessMode>
void test_constructor_with_no_init(const std::string& type_name,
                                   const std::string& access_mode_name) {
  const auto r = util::get_cts_object::range<Dimension>::get(1, 1, 1);
  const auto offset = sycl::id<Dimension>();
  const sycl::property_list prop_list(sycl::no_init);

  auto section_name = get_section_name<Dimension>(
      type_name, access_mode_name,
      "From buffer constructor with no_init property");

  SECTION(section_name) {
    auto construct_acc = [&prop_list](sycl::buffer<DataT, Dimension> data_buf) {
      return sycl::host_accessor<DataT, Dimension, AccessMode>(data_buf,
                                                               prop_list);
    };

    check_no_init_prop<AccType, DataT, Dimension, AccessMode>(construct_acc, r);
  }

  section_name = get_section_name<Dimension>(
      type_name, access_mode_name,
      "From buffer and range constructor with no_init property");

  SECTION(section_name) {
    auto construct_acc = [&prop_list,
                          r](sycl::buffer<DataT, Dimension> data_buf) {
      return sycl::host_accessor<DataT, Dimension, AccessMode>(data_buf, r,
                                                               prop_list);
    };

    check_no_init_prop<AccType, DataT, Dimension, AccessMode>(construct_acc, r);
  }

  section_name = get_section_name<Dimension>(
      type_name, access_mode_name,
      "From buffer,range and offset constructor with no_init property");

  SECTION(section_name) {
    auto construct_acc = [&prop_list, r,
                          offset](sycl::buffer<DataT, Dimension> data_buf) {
      return sycl::host_accessor<DataT, Dimension, AccessMode>(
          data_buf, r, offset, prop_list);
    };

    check_no_init_prop<AccType, DataT, Dimension, AccessMode>(construct_acc, r);
  }
}

template <typename DataT, int Dimension>
void test_exception(const std::string& type_name) {
  const auto r = util::get_cts_object::range<Dimension>::get(1, 1, 1);
  const auto offset = sycl::id<Dimension>();
  const sycl::property_list prop_list(sycl::no_init);

  auto section_name = get_section_name<Dimension>(
      type_name, "access_mode::read",
      "Expecting exception when attempting to construct accessor from buffer "
      "with no_init property and access_mode::read");
  SECTION(section_name) {
    auto construct_acc = [&prop_list](sycl::buffer<DataT, Dimension> data_buf) {
      sycl::host_accessor<DataT, Dimension, sycl::access_mode::read>(data_buf,
                                                                     prop_list);
    };
    check_no_init_prop_exception<AccType, DataT, Dimension>(construct_acc, r);
  }

  section_name = get_section_name<Dimension>(
      type_name, "access_mode::read",
      "Expecting exception when attempting to construct accessor from buffer "
      "and range with no_init property and access_mode::read");
  SECTION(section_name) {
    auto construct_acc = [&prop_list,
                          r](sycl::buffer<DataT, Dimension> data_buf) {
      sycl::host_accessor<DataT, Dimension, sycl::access_mode::read>(
          data_buf, r, prop_list);
    };
    check_no_init_prop_exception<AccType, DataT, Dimension>(construct_acc, r);
  }

  section_name = get_section_name<Dimension>(
      type_name, "access_mode::read",
      "Expecting exception when attempting to construct accessor from buffer, "
      "range and offset with no_init property and access_mode::read");
  SECTION(section_name) {
    auto construct_acc = [&prop_list, r,
                          offset](sycl::buffer<DataT, Dimension> data_buf) {
      sycl::host_accessor<DataT, Dimension, sycl::access_mode::read>(
          data_buf, r, offset, prop_list);
    };
    check_no_init_prop_exception<AccType, DataT, Dimension>(construct_acc, r);
  }
}

template <typename DataT, int Dimension, sycl::access_mode AccessMode>
void test_property_member_functions(const std::string& type_name,
                                    const std::string& access_mode_name) {
  const auto r = util::get_cts_object::range<Dimension>::get(1, 1, 1);
  const auto offset = sycl::id<Dimension>();
  const sycl::property_list prop_list(sycl::no_init);

  const auto construct_acc =
      [&prop_list](sycl::buffer<DataT, Dimension> data_buf) {
        return sycl::host_accessor<DataT, Dimension, AccessMode>(data_buf,
                                                                 prop_list);
      };

  auto section_name = get_section_name<Dimension>(
      type_name, access_mode_name, "has_property member function invocation");
  SECTION(section_name) {
    check_has_property_member_func<DataT, Dimension, sycl::property::no_init>(
        construct_acc, r);
  }
  section_name = get_section_name<Dimension>(
      type_name, access_mode_name, "get_property member function invocation");
  SECTION(section_name) {
    check_get_property_member_func<DataT, Dimension, sycl::property::no_init>(
        construct_acc, r);
  }
}

template <typename T, typename AccessModeT, typename DimensionT>
class run_tests_properties {
  static constexpr sycl::access_mode AccessMode = AccessModeT::value;
  static constexpr int Dimension = DimensionT::value;

 public:
  void operator()(const std::string& type_name,
                  const std::string& access_mode_name) {
    test_constructor_with_no_init<T, Dimension, AccessMode>(type_name,
                                                            access_mode_name);

    test_property_member_functions<T, Dimension, AccessMode>(type_name,
                                                             access_mode_name);

    // In order not to run again with same parameters
    if constexpr (AccessMode == sycl::access_mode::read) {
      test_exception<T, Dimension>(type_name);
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

    // To handle cases when class was called from functions
    // like for_all_types_vectors_marray or for_all_dev_copyable_containers.
    // This will wrap T to std::array<T,N> of T is array. Otherwise user will
    // see just type even if T was container for T
    auto actual_type_name = type_name_string<T>::get(type_name);

    for_all_combinations<run_tests_properties, const T>(
        access_modes, dimensions, actual_type_name);
  }
};
}  // namespace host_accessor_properties

#endif  // SYCL_CTS_HOST_ACCESSOR_PROPERTIES_H

/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for the properties of generic sycl::accessor
//
*******************************************************************************/
#ifndef SYCL_CTS_GENERIC_ACCESSOR_PROPERTIES_H
#define SYCL_CTS_GENERIC_ACCESSOR_PROPERTIES_H
#include "accessor_common.h"

namespace generic_accessor_properties {
using namespace sycl_cts;
using namespace accessor_tests_common;

constexpr accessor_type AccTypeT = accessor_type::generic_accessor;

template <typename DataT, int DimensionT, sycl::access_mode AccessModeT,
          sycl_stub::target TargetT>
void test_constructor_with_no_init(const std::string& type_name,
                                   const std::string& access_mode_name,
                                   const std::string& target_name) {
  auto r = util::get_cts_object::range<DimensionT>::get(1, 1, 1);
  auto offset = sycl::id<DimensionT>();
  const sycl::property_list prop_list(sycl::no_init);

  auto section_name = get_section_name<DimensionT>(
      type_name, access_mode_name, target_name,
      "From buffer constructor with no_init property");

  SECTION(section_name) {
    auto get_acc_functor = [&prop_list](
                               sycl::buffer<DataT, DimensionT>& data_buf,
                               sycl::handler& cgh) {
      return sycl_stub::accessor<DataT, DimensionT, AccessModeT, TargetT>(
          data_buf, cgh, prop_list);
    };

    check_no_init_prop<AccTypeT, DataT, DimensionT, AccessModeT, TargetT>(
        get_acc_functor);
  }

  section_name = get_section_name<DimensionT>(
      type_name, access_mode_name, target_name,
      "From buffer and range constructor with no_init property");

  SECTION(section_name) {
    auto get_acc_functor = [&prop_list, r](
                               sycl::buffer<DataT, DimensionT>& data_buf,
                               sycl::handler& cgh) {
      return sycl_stub::accessor<DataT, DimensionT, AccessModeT, TargetT>(
          data_buf, cgh, r, prop_list);
    };
    check_no_init_prop<AccTypeT, DataT, DimensionT, AccessModeT, TargetT>(
        get_acc_functor);
  }

  section_name = get_section_name<DimensionT>(
      type_name, access_mode_name, target_name,
      "From buffer,range and offset constructor with no_init property");

  SECTION(section_name) {
    auto get_acc_functor = [&prop_list, r, offset](
                               sycl::buffer<DataT, DimensionT>& data_buf,
                               sycl::handler& cgh) {
      return sycl_stub::accessor<DataT, DimensionT, AccessModeT, TargetT>(
          data_buf, cgh, r, offset, prop_list);
    };
    check_no_init_prop<AccTypeT, DataT, DimensionT, AccessModeT, TargetT>(
        get_acc_functor);
  }
}

template <typename DataT, int DimensionT, sycl_stub::target TargetT>
void test_exception(const std::string& type_name,
                    const std::string& target_name) {
  auto r = util::get_cts_object::range<DimensionT>::get(1, 1, 1);
  auto offset = sycl::id<DimensionT>();
  const sycl::property_list prop_list(sycl::no_init);

  auto section_name = get_section_name<DimensionT>(
      type_name, "access_mode::read", target_name,
      "Expecting exception when attempting to construct accessor from buffer "
      "with no_init property and access_mode::read");
  SECTION(section_name) {
    auto construct_acc = [&prop_list](
                             sycl::queue& queue,
                             sycl::buffer<DataT, DimensionT> data_buf) {
      queue
          .submit([&](sycl::handler& cgh) {
            sycl_stub::accessor<DataT, DimensionT, sycl::access_mode::read,
                                TargetT>(data_buf, cgh, prop_list);
          })
          .wait();
    };
    check_no_init_prop_exception<AccTypeT, DataT, DimensionT, TargetT>(
        construct_acc);
  }

  section_name = get_section_name<DimensionT>(
      type_name, "access_mode::read", target_name,
      "Expecting exception when attempting to construct accessor from buffer "
      "and range with no_init property and access_mode::read");
  SECTION(section_name) {
    auto construct_acc = [&prop_list, r](
                             sycl::queue& queue,
                             sycl::buffer<DataT, DimensionT> data_buf) {
      queue
          .submit([&](sycl::handler& cgh) {
            sycl_stub::accessor<DataT, DimensionT, sycl::access_mode::read,
                                TargetT>(data_buf, cgh, r, prop_list);
          })
          .wait();
    };
    check_no_init_prop_exception<AccTypeT, DataT, DimensionT, TargetT>(
        construct_acc);
  }

  section_name = get_section_name<DimensionT>(
      type_name, "access_mode::read", target_name,
      "Expecting exception when attempting to construct accessor from "
      "buffer, "
      "range and offset with no_init property and access_mode::read");
  SECTION(section_name) {
    auto construct_acc = [&prop_list, r, offset](
                             sycl::queue& queue,
                             sycl::buffer<DataT, DimensionT> data_buf) {
      queue
          .submit([&](sycl::handler& cgh) {
            sycl_stub::accessor<DataT, DimensionT, sycl::access_mode::read,
                                TargetT>(data_buf, cgh, r, offset, prop_list);
          })
          .wait();
    };
    check_no_init_prop_exception<AccTypeT, DataT, DimensionT, TargetT>(
        construct_acc);
  }
}

template <typename DataT, int DimensionT, sycl::access_mode AccessModeT,
          sycl_stub::target TargetT>
void test_property_member_functions(const std::string& type_name,
                                    const std::string& access_mode_name,
                                    const std::string& target_name) {
  auto r = util::get_cts_object::range<DimensionT>::get(1, 1, 1);
  auto offset = sycl::id<DimensionT>();
  const sycl::property_list prop_list(sycl::no_init);

  const auto construct_acc =
      [&prop_list](sycl::buffer<DataT, DimensionT> data_buf) {
        return sycl_stub::accessor<DataT, DimensionT, AccessModeT, TargetT>(
            data_buf, prop_list);
      };

  auto section_name =
      get_section_name<DimensionT>(type_name, access_mode_name, target_name,
                                   "has_property member function invocation");
  SECTION(section_name) {
    check_has_property_member_func<DataT, DimensionT, sycl::property::no_init>(
        construct_acc);
  }

  section_name =
      get_section_name<DimensionT>(type_name, access_mode_name, target_name,
                                   "get_property member function invocation");
  SECTION(section_name) {
    check_get_property_member_func<DataT, DimensionT, sycl::property::no_init>(
        construct_acc);
  }
}

template <typename T, typename AccessTypeT, typename TargetTypeT,
          typename DimensionTypeT>
class run_tests_properties {
  static constexpr sycl::access_mode AccessModeT = AccessTypeT::value;
  static constexpr int DimensionT = DimensionTypeT::value;
  static constexpr sycl_stub::target TargetT = TargetTypeT::value;

 public:
  void operator()(const std::string& type_name,
                  const std::string& access_mode_name,
                  const std::string& target_name) {
    test_constructor_with_no_init<T, DimensionT, AccessModeT, TargetT>(
        type_name, access_mode_name, target_name);

    test_property_member_functions<T, DimensionT, AccessModeT, TargetT>(
        type_name, access_mode_name, target_name);

    // In order not to run again with same parameters
    if constexpr (AccessModeT == sycl::access_mode::read) {
      test_exception<T, DimensionT, TargetT>(type_name, target_name);
    }
  }
};

template <typename T>
class run_generic_properties_tests {
 public:
  void operator()(const std::string& type_name) {
    // Type packs instances have to be const, otherwise for_all_combination
    // will not compile
    const auto access_modes = get_access_modes();
    const auto dimensions = get_dimensions();
    const auto targets = get_targets();
    const auto cur_type =
        named_type_pack<T>::generate(type_name_string<T>::get(type_name));

    for_all_combinations<run_tests_properties>(cur_type, access_modes, targets,
                                               dimensions);
  }
};
}  // namespace generic_accessor_properties

#endif  // SYCL_CTS_GENERIC_ACCESSOR_PROPERTIES_H

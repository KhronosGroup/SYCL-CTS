/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides common code for sycl::host_accessor correct linearization test
//  For multidimensional accessors the iterator linearizes the data according
//  to Section 3.11.1
//
*******************************************************************************/
#ifndef SYCL_CTS_HOST_ACCESSOR_LINEAR_COMMON_H
#define SYCL_CTS_HOST_ACCESSOR_LINEAR_COMMON_H
#include "accessor_common.h"

namespace host_accessor_linearization {
using namespace accessor_tests_common;

template <typename T, typename AccessT, typename DimensionT>
class run_linearization_tests {
  static constexpr sycl::access_mode AccessMode = AccessT::value;
  static constexpr int dims = DimensionT::value;
  using AccT = sycl::host_accessor<T, dims, AccessMode>;

 public:
  void operator()(const std::string &type_name,
                  const std::string &access_mode_name) {
    SECTION(get_section_name<dims>(type_name, access_mode_name, "")) {
      check_linearization<accessor_type::host_accessor, T, dims, AccessMode>();
    }
  }
};

template <typename T>
class run_host_linearization_for_type {
 public:
  void operator()(const std::string &type_name) {
    const auto access_modes = get_access_modes();
    const auto dimensions = integer_pack<2, 3>::generate_unnamed();
    auto actual_type_name = type_name_string<T>::get(type_name);

    for_all_combinations<run_linearization_tests, T>(access_modes, dimensions,
                                                     actual_type_name);

    // For covering const types
    actual_type_name = std::string("const ") + actual_type_name;
    // const T can be only with access_mode::read
    const auto read_only_acc_mode =
        value_pack<sycl::access_mode, sycl::access_mode::read>::generate_named(
            "access_mode::read");
    for_all_combinations<run_linearization_tests, const T>(
        read_only_acc_mode, dimensions, actual_type_name);
  }
};
}  // namespace host_accessor_linearization
#endif  // SYCL_CTS_HOST_ACCESSOR_LINEAR_COMMON_H

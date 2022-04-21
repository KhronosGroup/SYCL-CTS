/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  This file provides functions for tests on accessor exceptions.
//
*******************************************************************************/
#ifndef SYCL_CTS_ACCESSOR_EXCEPTIONS_H
#define SYCL_CTS_ACCESSOR_EXCEPTIONS_H

#include "../../util/usm_helper.h"
#include "accessor_common.h"
#include "catch2/catch_test_macros.hpp"

#include <type_traits>

namespace accessor_exceptions_test {

using generic_accessor = std::integral_constant<
    accessor_tests_common::accessor_type,
    accessor_tests_common::accessor_type::generic_accessor>;
using local_accessor = std::integral_constant<
    accessor_tests_common::accessor_type,
    accessor_tests_common::accessor_type::local_accessor>;
using host_accessor =
    std::integral_constant<accessor_tests_common::accessor_type,
                           accessor_tests_common::accessor_type::host_accessor>;

/**
 * @brief Function helps to verify that constructors accessor really thrown
 *        exceptions.
 * @tparam AccType Current type of the accessor: generic_accessor,
 *         local_accessor, or host_accessor
 * @tparam DataT Current data type
 * @tparam Dimension Dimension size
 * @tparam Target Current target
 * @tparam GetAccFunctorT Type of the functor that will be use to interact with
 *         accessor
 */
template <accessor_tests_common::accessor_type AccType, typename DataT,
          int Dimension, sycl::target Target = sycl::target::device,
          typename GetAccFunctorT>
void check_exception(GetAccFunctorT construct_acc) {
  auto queue = util::get_cts_object::queue();
  DataT some_data(expected_val);
  auto r = util::get_cts_object::range<Dimension>::get(1, 1, 1);
  {
    sycl::buffer<DataT, Dimension> data_buf(&some_data, r);

    if constexpr (AccType == accessor_type::generic_accessor) {
      auto action = [&] {
        // Submit an obtained lambda, then wait wait for lambda execution, then
        // throw an exception if it should be thrown.
        queue.submit([&](sycl::handler& cgh) { construct_acc(cgh, data_buf); })
            .wait_and_throw();
      };
      CHECK_THROWS_MATCHES(
          action, sycl::exception,
          sycl_cts::util::equals_exception(sycl::errc::invalid));
    } else if constexpr (AccType ==
                         accessor_tests_common::accessor_type::host_accessor) {
      auto action = [&] { construct_acc(data_buf); };
      CHECK_THROWS_MATCHES(
          action, sycl::exception,
          sycl_cts::util::equals_exception(sycl::errc::invalid));
    } else if constexpr (AccType ==
                         accessor_tests_common::accessor_type::local_accessor) {
      static_cast<void>(data_buf);
      auto action = [&] { construct_acc(queue); };
      CHECK_THROWS_MATCHES(
          action, sycl::exception,
          sycl_cts::util::equals_exception(sycl::errc::kernel_argument));
    } else {
      static_assert(AccType != AccType, "Unexpected accessor type.");
    }
  }
}

/**
 * @brief Provides functor that lets verify that local_accessor really thrown
 *        exception.
 * @tparam AccT Current type of the accessor: generic_accessor,
 *         local_accessor, or host_accessor
 * @tparam DataT Current data type
 * @tparam DimensionT Dimension size
 * @param type_name Current data type string representation
 */
template <typename AccT, typename DataT, typename DimensionT>
class test_exception_for_local_acc {
  static constexpr int Dimension = DimensionT::value;
  static constexpr auto AccType = AccT::value;

 public:
  void operator()(const std::string& type_name) {
    auto section_name =
        "Expecting exception when attempting to invoke local_accessor via "
        "single_task for " +
        type_name + " data type.";
    SECTION(section_name) {
      auto construct_acc = [](sycl::queue& queue) {
        // Use a variable to avoid device code optimisation.
        auto is_empty =
            usm_helper::allocate_usm_memory<sycl::usm::alloc::shared, bool>(
                queue);
        queue
            .submit([&](sycl::handler& cgh) {
              auto is_empty_ptr = is_empty.get();
              sycl::local_accessor<DataT, Dimension> local_acc(cgh);
              cgh.single_task([=](sycl::kernel_handler cgh) {
                // Some interactions to avoid device code optimisation.
                *is_empty_ptr = local_acc.empty();
              });
            })
            .wait();
      };
      check_exception<AccTypeT::value, DataT, Dimension>(construct_acc);
    }
  }
};

/**
 * @brief Provides functor that lets verify that host_accessor really thrown
 *        exception.
 * @tparam AccT Current type of the accessor: generic_accessor,
 *         local_accessor, or host_accessor
 * @tparam DataT Current data type
 * @tparam AccessModeT Field of sycl::access_mode enumeration
 * @tparam DimensionT Dimension size
 * @tparam TargetT Current target
 * @param type_name Current data type string representation
 * @param access_mode_name Current access mode string representation
 * @param target_name Current target string representation
 */
template <typename AccT, typename DataT, typename AccessModeT,
          typename DimensionT, typename TargetT>
class test_exception_for_host_acc {
  static constexpr sycl::access_mode AccessMode = AccessModeT::value;
  static constexpr int Dimension = DimensionT::value;
  static constexpr sycl::target Target = TargetT::value;
  static constexpr auto AccType = AccT::value;

 public:
  void operator()(const std::string& type_name,
                  const std::string& access_mode_name,
                  const std::string& target_name) {
    auto great_range = util::get_cts_object::range<Dimension>::get(10, 10, 10);
    auto default_range = util::get_cts_object::range<Dimension>::get(1, 1, 1);
    auto id = util::get_cts_object::id<Dimension>::get(1, 1, 1);

    auto section_name = get_section_name<Dimension>(
        type_name, access_mode_name, target_name,
        "Expecting exception when attempting to construct host_accessor from "
        "buffer and range. In case, the range exceeds the range of buffer "
        "in any dimension.");
    SECTION(section_name) {
      auto construct_acc =
          [&great_range](sycl::buffer<DataT, Dimension> data_buf) {
            sycl::host_accessor<DataT, Dimension, AccessMode>(data_buf,
                                                              great_range);
          };
      check_exception<AccType, DataT, Dimension, Target>(construct_acc);
    }

    section_name = get_section_name<Dimension>(
        type_name, access_mode_name, target_name,
        "Expecting exception when attempting to construct host_accessor from "
        "buffer, range and tag. In case, the range exceeds the range of buffer "
        "in any dimension.");
    SECTION(section_name) {
      auto construct_acc =
          [&great_range](sycl::buffer<DataT, Dimension> data_buf) {
            sycl::host_accessor<DataT, Dimension>(data_buf, great_range,
                                                  sycl::read_only);
          };
      check_exception<AccType, DataT, Dimension, Target>(construct_acc);
    }

    section_name = get_section_name<Dimension>(
        type_name, access_mode_name, target_name,
        "Expecting exception when attempting to construct host_accessor from "
        "buffer, and range and id. In case, when the sum of range and offset "
        "exceeds the range of buffer in any dimension.");
    SECTION(section_name) {
      auto construct_acc = [&default_range,
                            id](sycl::buffer<DataT, Dimension> data_buf) {
        sycl::host_accessor<DataT, Dimension, AccessMode>(data_buf,
                                                          default_range, id);
      };
      check_exception<AccType, DataT, Dimension, Target>(construct_acc);
    }

    section_name = get_section_name<Dimension>(
        type_name, access_mode_name, target_name,
        "Expecting exception when attempting to construct host_accessor from "
        "buffer, and range, id and tag. In case, when the sum of range and "
        "offset exceeds the range of buffer in any dimension.");
    SECTION(section_name) {
      auto construct_acc = [&default_range,
                            id](sycl::buffer<DataT, Dimension> data_buf) {
        sycl::host_accessor<DataT, Dimension>(data_buf, default_range, id,
                                              sycl::read_only);
      };
      check_exception<AccType, DataT, Dimension, Target>(construct_acc);
    }
  }
};

/**
 * @brief Provides functor that lets verify that generic accessor really thrown
 *        exception.
 * @tparam AccT Current type of the accessor: generic_accessor,
 *         local_accessor, or host_accessor
 * @tparam DataT Current data type
 * @tparam AccessModeT Field of sycl::access_mode enumeration
 * @tparam DimensionT Dimension size
 * @tparam TargetT Current target
 * @param type_name Current data type string representation
 * @param access_mode_name Current access mode string representation
 * @param target_name Current target string representation
 */
template <typename AccT, typename DataT, typename AccessModeT,
          typename DimensionT, typename TargetT>
class test_exception_for_generic_acc {
  static constexpr sycl::access_mode AccessMode = AccessModeT::value;
  static constexpr int Dimension = DimensionT::value;
  static constexpr sycl::target Target = TargetT::value;
  static constexpr auto AccType = AccT::value;

 public:
  void operator()(const std::string& type_name,
                  const std::string& access_mode_name,
                  const std::string& target_name) {
    auto great_range = util::get_cts_object::range<Dimension>::get(10, 10, 10);
    auto default_range = util::get_cts_object::range<Dimension>::get(1, 1, 1);
    auto id = util::get_cts_object::id<Dimension>::get(1, 1, 1);

    auto section_name = get_section_name<Dimension>(
        type_name, access_mode_name, target_name,
        "Expecting exception when attempting to construct host_accessor from "
        "buffer, and range that range and offset with no_init property and "
        "access_mode::read");
    SECTION(section_name) {
      auto construct_acc = [&great_range](
                               sycl::handler& cgh,
                               sycl::buffer<DataT, Dimension> data_buf) {
        sycl::host_accessor<DataT, Dimension, AccessMode>(data_buf,
                                                          great_range);
      };
      check_exception<AccType, DataT, Dimension, Target>(construct_acc);
    }

    section_name = get_section_name<Dimension>(
        type_name, access_mode_name, target_name,
        "Expecting exception when attempting to construct accessor from "
        "buffer, tag and range. In case, the range exceeds the range of buffer "
        "in any dimension.");
    SECTION(section_name) {
      auto construct_acc = [&great_range](
                               sycl::handler& cgh,
                               sycl::buffer<DataT, Dimension> data_buf) {
        sycl::accessor<DataT, Dimension>(data_buf, great_range,
                                         sycl::read_only);
      };
      check_exception<AccType, DataT, Dimension, Target>(construct_acc);
    }

    section_name =
        get_section_name<Dimension>(type_name, access_mode_name, target_name,
                                    "Expecting exception when attempting to "
                                    "construct accessor from buffer, "
                                    "and range and id. In case, when the sum "
                                    "of range and offset exceeds the "
                                    "range of buffer in any dimension.");
    SECTION(section_name) {
      auto construct_acc = [&default_range, id](
                               sycl::handler& cgh,
                               sycl::buffer<DataT, Dimension> data_buf) {
        sycl::accessor<DataT, Dimension, AccessMode, Target>(data_buf,
                                                             default_range, id);
      };
      check_exception<AccType, DataT, Dimension, Target>(construct_acc);
    }

    section_name = get_section_name<Dimension>(
        type_name, access_mode_name, target_name,
        "Expecting exception when attempting to construct accessor from "
        "buffer, "
        "tag, range and id. In case, when the sum of range and offset exceeds "
        "the range of buffer in any dimension.");
    SECTION(section_name) {
      auto construct_acc = [&default_range, id](
                               sycl::handler& cgh,
                               sycl::buffer<DataT, Dimension> data_buf) {
        sycl::accessor<DataT, Dimension>(data_buf, default_range, id,
                                         sycl::read_only);
      };
      check_exception<AccType, DataT, Dimension, Target>(construct_acc);
    }

    section_name = get_section_name<Dimension>(
        type_name, access_mode_name, target_name,
        "Expecting exception when attempting to construct accessor from "
        "buffer, handler and range. In case, the range exceeds the range of "
        "buffer in any dimension");
    SECTION(section_name) {
      auto construct_acc = [&great_range](
                               sycl::handler& cgh,
                               sycl::buffer<DataT, Dimension> data_buf) {
        sycl::accessor<DataT, Dimension, AccessMode, Target>(data_buf, cgh,
                                                             great_range);
      };
      check_exception<AccType, DataT, Dimension, Target>(construct_acc);
    }

    section_name = get_section_name<Dimension>(
        type_name, access_mode_name, target_name,
        "Expecting exception when attempting to construct accessor from "
        "buffer, handler, range and tag. In case, the range exceeds the range "
        "of "
        "buffer in any dimension.");
    SECTION(section_name) {
      auto construct_acc = [&great_range](
                               sycl::handler& cgh,
                               sycl::buffer<DataT, Dimension> data_buf) {
        sycl::accessor<DataT, Dimension>(data_buf, cgh, great_range,
                                         sycl::read_only);
      };
      check_exception<AccType, DataT, Dimension, Target>(construct_acc);
    }

    section_name = get_section_name<Dimension>(
        type_name, access_mode_name, target_name,
        "Expecting exception when attempting to construct accessor from "
        "buffer, handler, range and tag. In case, when the sum of range and "
        "offset exceeds the range of buffer in any dimension.");
    SECTION(section_name) {
      auto construct_acc = [&default_range, id](
                               sycl::handler& cgh,
                               sycl::buffer<DataT, Dimension> data_buf) {
        sycl::accessor<DataT, Dimension, AccessMode, Target>(data_buf, cgh,
                                                             default_range, id);
      };
      check_exception<AccType, DataT, Dimension, Target>(construct_acc);
    }
  }
};

/**
 * @brief Struct that runs test with different input parameters types
 * @tparam AccT Current type of the accessor: generic_accessor,
 *         local_accessor, or host_accessor
 * @tparam T Current data type
 * @param type_name Current data type string representation
 * @param access_mode_name Current access mode string representation
 * @param target_name Current target string representation
 */
template <typename T, typename AccT>
class run_tests_with_types {
 public:
  void operator()(const std::string& type_name) {
    // Type packs instances have to be const, otherwise for_all_combination
    // will not compile
    const auto access_modes = get_access_modes();
    const auto dimensions = get_all_dimensions();
    const auto targets = get_targets();

    constexpr accessor_tests_common::accessor_type acc_type = AccT::value;
    if constexpr (acc_type ==
                  accessor_tests_common::accessor_type::generic_accessor) {
      for_all_combinations<test_exception_for_generic_acc, AccT, T>(
          access_modes, dimensions, targets, type_name);
    } else if constexpr (acc_type ==
                         accessor_tests_common::accessor_type::host_accessor) {
      for_all_combinations<test_exception_for_host_acc, AccT, T>(
          access_modes, dimensions, targets, type_name);
    } else if constexpr (acc_type ==
                         accessor_tests_common::accessor_type::local_accessor) {
      for_all_combinations<test_exception_for_local_acc, AccT, T>(dimensions,
                                                                  type_name);
    }
  }
};

}  // namespace accessor_exceptions_test

#endif  // SYCL_CTS_ACCESSOR_EXCEPTIONS_H

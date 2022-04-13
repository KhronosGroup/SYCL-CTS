/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Common functions for the accessor tests.
//
*******************************************************************************/

#ifndef SYCL_CTS_ACCESSOR_COMMON_H
#define SYCL_CTS_ACCESSOR_COMMON_H

#include "../../util/sycl_exceptions.h"
#include "../common/common.h"
#include "../common/type_coverage.h"
#include "../common/value_helper.h"

#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers.hpp"

namespace accessor_tests_common {
using namespace sycl_cts;

constexpr int expected_val = 42;
constexpr int changed_val = 1;

/**
 * @brief Function helps to get string section name that will contain template
 * parameters and function arguments
 *
 * @tparam DimensionT Integer representing dimension
 * @param type_name String with name of the testing type
 * @param access_mode_name String with name of the testing access mode
 * @param target_name String with name of the testing target
 * @param section_description String with human-readable description of the test
 * @return std::string String with name for section
 */
template <int DimensionT>
inline std::string get_section_name(const std::string& type_name,
                                    const std::string& access_mode_name,
                                    const std::string& target_name,
                                    const std::string& section_description) {
  std::string name = "Test ";
  name += section_description;
  name += " with parameters: <";
  name += type_name;
  name += "><";
  name += access_mode_name;
  name += "><";
  name += target_name;
  name += "><";
  name += std::to_string(DimensionT) + ">";
  return name;
}

/**
 * @brief Function helps to get string section name that will contain template
 * parameters and function arguments
 *
 * @tparam DimensionT Integer representing dimension
 * @param type_name String with name of the testing type
 * @param access_mode_name String with name of the testing access mode
 * @param target_name String with name of the testing target
 * @param section_description String with human-readable description of the test
 * @return std::string String with name for section
 */
template <int DimensionT>
inline std::string get_section_name(const std::string& type_name,
                                    const std::string& access_mode_name,
                                    const std::string& section_description) {

  std::string name = "Test ";
  name += section_description;
  name += " with parameters: <";
  name += type_name;
  name += "><";
  name += access_mode_name;
  name += "><";
  name += std::to_string(DimensionT) + ">";
  return name;
}

/**
 * @brief Factory function for getting type_pack with fp16 type
 */
inline auto get_fp16_type() {
  static const auto types = named_type_pack<sycl::half>::generate("sycl::half");
  return types;
}

/**
 * @brief Factory function for getting type_pack with fp64 type
 */
inline auto get_fp64_type() {
  static const auto types = named_type_pack<double>::generate("double");
  return types;
}

/**
 * @brief Factory function for getting type_pack with all generic types
 */
inline auto get_full_conformance_type_pack() {
  static const auto types =
      named_type_pack<bool, char, signed char, unsigned char, short int,
                      unsigned short int, int, unsigned int, long int,
                      unsigned long int, long long int, unsigned long long int,
                      float>::generate("bool", "char", "signed char",
                                       "unsigned char", "short int",
                                       "unsigned short int", "int",
                                       "unsigned int", "long int",
                                       "unsigned long int", "long long int",
                                       "unsigned long long int", "float");
  return types;
}

/**
 * @brief Factory function for getting type_pack with generic types
 */
inline auto get_lightweight_type_pack() {
  static const auto types =
      named_type_pack<bool, int, float>::generate("bool", "int", "float");
  return types;
}

/**
 * @brief Factory function for getting type_pack with access modes values
 */
inline auto get_access_modes() {
  static const auto access_modes = value_pack<
      sycl::access_mode, sycl::access_mode::read, sycl::access_mode::write,
      sycl::access_mode::read_write>::generate_named("access_mode::read",
                                                     "access_mode::write",
                                                     "access_mode::read_write");
  return access_modes;
}

/**
 * @brief Factory function for getting type_pack with dimensions values
 */
inline auto get_dimensions() {
  static const auto dimensions = integer_pack<1, 2, 3>::generate_unnamed();
  return dimensions;
}

/**
 * @brief Factory function for getting type_pack with target values
 */
inline auto get_targets() {
  static const auto targets =
      value_pack<sycl::target, sycl::target::device,
                 sycl::target::host_task>::generate_named("target::device",
                                                          "target::host_task");
  return targets;
}

/**
 * @brief Function helps to generate type_pack with sycl::vec of all supported
 * sizes
 */
template <typename T, typename StrNameType>
inline auto add_vectors_to_type_pack(StrNameType type_name) {
  return named_type_pack<
      T, sycl::vec<T, 1>, sycl::vec<T, 2>, sycl::vec<T, 4>, sycl::vec<T, 8>,
      sycl::vec<T, 16>>::generate(type_name, "vec<" + type_name + ", 1>",
                                  "vec<" + type_name + ", 2>",
                                  "vec<" + type_name + ", 4>",
                                  "vec<" + type_name + ", 8>",
                                  "vec<" + type_name + ", 16>");
}

/**
 * @brief Function helps to get TagT corresponding to ModeT and TargetT template
 * parameters
 */
template <sycl::access_mode ModeT, sycl::target TargetT>
auto get_tag() {
  if constexpr (TargetT == sycl::target::device) {
    if constexpr (ModeT == sycl::access_mode::read) {
      return sycl::read_only;
    } else if constexpr (ModeT == sycl::access_mode::write) {
      return sycl::write_only;
    } else if constexpr (ModeT == sycl::access_mode::read_write) {
      return sycl::read_write;
    }
  } else if constexpr (TargetT == sycl::target::host_task) {
    if constexpr (ModeT == sycl::access_mode::read) {
      return sycl::read_only_host_task;
    } else if constexpr (ModeT == sycl::access_mode::write) {
      return sycl::write_only_host_task;
    } else if constexpr (ModeT == sycl::access_mode::read_write) {
      return sycl::read_write_host_task;
    }
  }
}

/**
 * @brief Enum class for accessor type specification
 */
enum class accessor_type {
  generic_accessor,
  local_accessor,
  host_accessor,
};

/**
 * @brief Common function that check default constructor post-conditions, and
 * store result in res_acc
 *
 * @tparam TestingAccT Type of testing accessor
 * @tparam ResultAccT  Type of result accessor
 * @param testing_acc Instance of TestingAccT that were constructed with default
 * constructor
 * @param res_acc Instance of result accessor
 */
template <typename TestingAccT, typename ResultAccT>
void check_def_constructor_post_conditions(TestingAccT testing_acc,
                                           ResultAccT res_acc) {
  size_t res_i = 0;
  // (empty() == true)
  res_acc[res_i++] = testing_acc.empty() == true;

  // All size queries return 0
  res_acc[res_i++] = testing_acc.byte_size() == 0;
  res_acc[res_i++] = testing_acc.size() == 0;
  res_acc[res_i++] = testing_acc.max_size() == 0;

  // The only iterator that can be obtained is nullptr
  res_acc[res_i++] = testing_acc.begin() == testing_acc.end();
  res_acc[res_i++] = testing_acc.cbegin() == testing_acc.cend();
  res_acc[res_i++] = testing_acc.rbegin() == testing_acc.rend();
  res_acc[res_i++] = testing_acc.crbegin() == testing_acc.crend();
}

/**
 * @brief Common function that constructs TestingAccT with default constructor
 *and checks post-conditions
 *
 * @tparam TestingAccT Type of testing accessor
 * @tparam TargetT Target of TestingAccT
 */
template <accessor_type AccTypeT, typename DataT, int DimensionT,
          sycl::access_mode AccessModeT = sycl::access_mode::read_write,
          sycl::target TargetT = sycl::target::device, typename GetAccFunctorT>
void check_def_constructor(GetAccFunctorT get_accessor_functor) {
  auto queue = util::get_cts_object::queue();
  sycl::range<1> r(1);
  const size_t conditions_checks_size = 8;
  bool conditions_check[conditions_checks_size]{false};
  {
    sycl::buffer res_buf(conditions_check, sycl::range(conditions_checks_size));

    queue
        .submit([&](sycl::handler& cgh) {
          sycl::accessor res_acc(res_buf);
          auto acc = get_accessor_functor();
          if constexpr (TargetT == sycl::target::host_task) {
            cgh.host_task(
                [=] { check_def_constructor_post_conditions(acc, res_acc); });
          } else if constexpr (TargetT == sycl::target::device) {
            cgh.parallel_for_work_group(r, [acc, res_acc](sycl::group<1>) {
              check_def_constructor_post_conditions(acc, res_acc);
            });
          }
        })
        .wait_and_throw();
  }

  for (size_t i = 0; i < conditions_checks_size; i++) {
    CHECK(conditions_check[i]);
  }
}

/**
 * @brief Function that tries to read or/and write depending on AccessModeT
 * parameter. Results of compare will be stored in res_acc
 *
 * @tparam AccT Type of testing accessor
 * @tparam ResultAccT Type of accessor for storing result
 * @param testing_acc Instance of sycl::accessor to read/write
 * @param res_acc Accessor for storing result
 */
template <typename DataT, sycl::access_mode AccessModeT, typename AccT,
          typename ResultAccT>
void read_write_zero_dim_acc(AccT testing_acc, ResultAccT res_acc) {
  DataT other_data(expected_val);

  if constexpr (AccessModeT != sycl::access_mode::write) {
    DataT acc_ref(testing_acc);
    res_acc[0] = value_helper::compare_vals(acc_ref, other_data);
  }
  if constexpr (AccessModeT != sycl::access_mode::read) {
    DataT acc_ref(testing_acc);
    value_helper::change_val(acc_ref, changed_val);
  }
}

/**
 * @brief Function helps to check zero dimension constructor of accessor
 *
 * @tparam GetAccFunctorT Type of functor that constructs testing accessor
 */
template <accessor_type AccTypeT, typename DataT, sycl::access_mode AccessModeT,
          sycl::target TargetT, typename GetAccFunctorT>
void check_zero_dim_constructor(GetAccFunctorT get_accessor_functor) {
  auto queue = util::get_cts_object::queue();
  sycl::range<1> r(1);
  DataT some_data(expected_val);

  bool compare_res = false;
  {
    sycl::buffer res_buf(&compare_res, r);
    sycl::buffer<DataT, 1> data_buf(&some_data, r);

    queue
        .submit([&](sycl::handler& cgh) {
          sycl::accessor res_acc(res_buf);

          auto acc = get_accessor_functor(data_buf, cgh);
          if constexpr (TargetT == sycl::target::host_task) {
            cgh.host_task([=] {
              read_write_zero_dim_acc<DataT, AccessModeT>(acc, res_acc);
            });
          } else if constexpr (TargetT == sycl::target::device) {
            cgh.parallel_for_work_group(r, [=](sycl::group<1>) {
              read_write_zero_dim_acc<DataT, AccessModeT>(acc, res_acc);
            });
          }
        })
        .wait_and_throw();
  }

  if constexpr (AccessModeT != sycl::access_mode::write) {
    CHECK(compare_res);
  }
  if constexpr (AccessModeT != sycl::access_mode::read) {
    CHECK(value_helper::compare_vals(some_data, changed_val));
  }
}

/**
 * @brief Function that tries to read or/and write depending on AccessModeT
 * parameter. Results of compare will be stored in res_acc
 *
 * @tparam AccT Type of testing accessor
 * @tparam ResultAccT Type of accessor for storing result
 * @param testing_acc Instance of sycl::accessor to read/write
 * @param res_acc Accessor for storing result
 */
template <typename DataT, int DimensionT, sycl::access_mode AccessModeT,
          typename AccT, typename ResultAccT>
void read_write_acc(AccT testing_acc, ResultAccT res_acc) {
  DataT other_data(expected_val);
  auto id = util::get_cts_object::id<DimensionT>::get(0, 0, 0);

  if constexpr (AccessModeT != sycl::access_mode::write) {
    res_acc[0] = value_helper::compare_vals(testing_acc[id], other_data);
  }
  if constexpr (AccessModeT != sycl::access_mode::read) {
    value_helper::change_val(testing_acc[id], changed_val);
  }
}

/**
 * @brief Function helps to check common constructor of accessor
 *
 * @tparam GetAccFunctorT Type of functor that constructs testing accessor
 */
template <accessor_type AccTypeT, typename DataT, int DimensionT,
          sycl::access_mode AccessModeT, sycl::target TargetT,
          typename GetAccFunctorT>
void check_common_constructor(GetAccFunctorT get_accessor_functor) {
  auto queue = util::get_cts_object::queue();
  bool compare_res = false;
  DataT some_data(expected_val);
  auto r = util::get_cts_object::range<DimensionT>::get(1, 1, 1);
  {
    sycl::buffer res_buf(&compare_res, sycl::range(1));
    sycl::buffer<DataT, DimensionT> data_buf(&some_data, r);

    queue
        .submit([&](sycl::handler& cgh) {
          sycl::accessor res_acc(res_buf);
          auto acc = get_accessor_functor(data_buf, cgh);

          if (acc.is_placeholder()) {
            cgh.require(acc);
          }

          if (TargetT == sycl::target::host_task) {
            cgh.host_task([=] {
              read_write_acc<DataT, DimensionT, AccessModeT>(acc, res_acc);
            });
          } else if (TargetT == sycl::target::device) {
            cgh.parallel_for_work_group(sycl::range(1), [=](sycl::group<1>) {
              read_write_acc<DataT, DimensionT, AccessModeT>(acc, res_acc);
            });
          }
        })
        .wait_and_throw();
  }

  if constexpr (AccessModeT != sycl::access_mode::write) {
    CHECK(compare_res);
  }
  if constexpr (AccessModeT != sycl::access_mode::read) {
    CHECK(value_helper::compare_vals(some_data, changed_val));
  }
}

/**
 * @brief Function helps to check if passing of a placeholder accessor triggers
 * the exception
 *
 * @tparam GetAccFunctorT Type of functor that constructs testing accessor
 */
template <accessor_type AccTypeT, typename DataT, int DimensionT,
          sycl::access_mode AccessModeT, sycl::target TargetT,
          typename GetAccFunctorT>
void check_placeholder_accessor_exception(GetAccFunctorT get_accessor_functor) {
  auto queue = util::get_cts_object::queue();
  DataT some_data(expected_val);
  bool is_placeholder = false;
  auto r = util::get_cts_object::range<DimensionT>::get(1, 1, 1);
  {
    sycl::buffer<DataT, DimensionT> data_buf(&some_data, r);

    auto action = [&] {
      queue
          .submit([&](sycl::handler& cgh) {
            auto acc = get_accessor_functor(data_buf);
            is_placeholder = acc.is_placeholder();
            if (TargetT == sycl::target::host_task) {
              cgh.host_task([=] {});
            } else if (TargetT == sycl::target::device) {
              cgh.parallel_for_work_group(sycl::range(1),
                                          [=](sycl::group<1>) {});
            }
          })
          .wait_and_throw();
    };
    CHECK(is_placeholder);
    INFO(
        "Implementation has to throw a sycl::exception with "
        "sycl::errc::kernel_argument when a placeholder accessor is passed to "
        "the command");
    CHECK_THROWS_MATCHES(
        action, sycl::exception,
        sycl_cts::util::equals_exception(sycl::errc::kernel_argument));
  }
}

/**
 * @brief Function checks common buffer and local accessor member functions
 */
template <typename AccT, int dims>
void test_accessor_methods_common(const AccT& accessor,
                                  const size_t expected_byte_size,
                                  const size_t expected_size,
                                  const sycl::range<dims>& expected_range) {
  {
    INFO("check byte_size() method");
    auto acc_byte_size = accessor.byte_size();
    STATIC_CHECK(
        std::is_same_v<decltype(acc_byte_size), typename AccT::size_type>);
    CHECK(acc_byte_size == expected_byte_size);
  }
  {
    INFO("check size() method");
    auto acc_size = accessor.size();
    STATIC_CHECK(std::is_same_v<decltype(acc_size), typename AccT::size_type>);
    CHECK(acc_size == expected_size);
  }
  {
    INFO("check max_size() method");
    auto acc_max_size = accessor.max_size();
    STATIC_CHECK(
        std::is_same_v<decltype(acc_max_size), typename AccT::size_type>);
  }
  {
    INFO("check empty() method");
    auto acc_empty = accessor.empty();
    STATIC_CHECK(std::is_same_v<decltype(acc_empty), bool>);
    CHECK(acc_empty == (expected_size == 0));
  }
  {
    INFO("check get_range() method");
    auto acc_range = accessor.get_range();
    STATIC_CHECK(std::is_same_v<decltype(acc_range), sycl::range<dims>>);
    CHECK(acc_range == expected_range);
  }
}

/**
 * @brief Function checks common buffer and local accessor member types
 */
template <typename T, typename AccT, sycl::access_mode mode>
void test_accessor_types_common() {
  if constexpr (mode != sycl::access_mode::read) {
    STATIC_CHECK(std::is_same_v<typename AccT::value_type, T>);
  } else {
    STATIC_CHECK(std::is_same_v<typename AccT::value_type, const T>);
  }
  STATIC_CHECK(
      std::is_same_v<typename AccT::reference, typename AccT::value_type&>);
  STATIC_CHECK(std::is_same_v<typename AccT::const_reference, const T&>);
  STATIC_CHECK(std::is_same_v<typename AccT::size_type, size_t>);
}

template <typename T, typename AccT, int dims>
T& get_subscript_overload(const AccT& accessor, size_t index) {
  if constexpr (dims == 1) return accessor[index];
  if constexpr (dims == 2) return accessor[index][index];
  if constexpr (dims == 3) return accessor[index][index][index];
}

}  // namespace accessor_tests_common

#endif  // SYCL_CTS_ACCESSOR_COMMON_H

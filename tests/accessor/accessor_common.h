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

//  Common functions for the accessor tests.

#ifndef SYCL_CTS_ACCESSOR_COMMON_H
#define SYCL_CTS_ACCESSOR_COMMON_H

#include "../../util/sycl_exceptions.h"
#include "../common/common.h"
#include "../common/section_name_builder.h"
#include "../common/value_operations.h"
#include "../common/once_per_unit.h"

// FIXME: re-enable when marrray is implemented in hipsycl
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
#include "../common/type_coverage.h"
#endif

#include <catch2/matchers/catch_matchers.hpp>

namespace accessor_tests_common {
using namespace sycl_cts;

constexpr int expected_val = 42;
constexpr int changed_val = 1;

/**
 * @brief Enum class for accessor type specification
 */
enum class accessor_type {
  generic_accessor,  // Buffer accessor for commands (Paragraph 4.7.6.9. of the
                     // spec)
  local_accessor,
  host_accessor,
};

}  // namespace accessor_tests_common

namespace Catch {
template <>
struct StringMaker<accessor_tests_common::accessor_type> {
  using type = accessor_tests_common::accessor_type;
  static std::string convert(type value) {
    switch (value) {
      case type::generic_accessor:
        return "sycl::accessor";
      case type::local_accessor:
        return "sycl::local_accessor";
      case type::host_accessor:
        return "sycl::host_accessor";
      default:
        return "unknown accessor type";
    }
  }
};
}  // namespace Catch

namespace accessor_tests_common {

/**
 * @brief Function helps to get string section name that will contain template
 * parameters and function arguments
 *
 * @tparam Dimension Integer representing dimension
 * @param type_name String with name of the testing type
 * @param access_mode_name String with name of the testing access mode
 * @param target_name String with name of the testing target
 * @param section_description String with human-readable description of the test
 * @return std::string String with name for section
 */
template <int Dimension>
inline std::string get_section_name(const std::string& type_name,
                                    const std::string& access_mode_name,
                                    const std::string& target_name,
                                    const std::string& section_description) {
  return section_name(section_description)
      .with("T", type_name)
      .with("access mode", access_mode_name)
      .with("target", target_name)
      .with("dimension", Dimension)
      .create();
}

/**
 * @brief Function helps to get string section name that will contain template
 * parameters and function arguments
 *
 * @tparam Dimension Integer representing dimension
 * @param type_name String with name of the testing type
 * @param access_mode_name String with name of the testing access mode
 * @param target_name String with name of the testing target
 * @param section_description String with human-readable description of the test
 * @return std::string String with name for section
 */
template <int Dimension>
inline std::string get_section_name(const std::string& type_name,
                                    const std::string& access_mode_name,
                                    const std::string& section_description) {
  return section_name(section_description)
      .with("T", type_name)
      .with("access mode", access_mode_name)
      .with("dimension", Dimension)
      .create();
}

/**
 * @brief Function helps to get string section name that will contain template
 * parameters and function arguments
 *
 * @tparam Dimension Integer representing dimension
 * @param type_name String with name of the testing type
 * @param section_description String with human-readable description of the test
 * @return std::string String with name for section
 */
template <int Dimension>
inline std::string get_section_name(const std::string& type_name,
                                    const std::string& section_description) {
  return section_name(section_description)
      .with("T", type_name)
      .with("dimension", Dimension)
      .create();
}

// FIXME: re-enable when marrray is implemented in hipsycl and type_coverage is
// enabled
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
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
 * @brief Factory function for getting type_pack with types that depends on full
 *        conformance mode enabling status
 * @return lightweight or full named_type_pack
 */
inline auto get_conformance_type_pack() {
#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
  return get_full_conformance_type_pack();
#else
  return get_lightweight_type_pack();
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
}

/**
 * @brief Factory function for getting type_pack with access modes values
 */
inline auto get_access_modes() {
  static const auto access_modes =
      value_pack<sycl::access_mode, sycl::access_mode::read,
                 sycl::access_mode::write,
                 sycl::access_mode::read_write>::generate_named();
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
 * @brief Factory function for getting type_pack with all (including zero)
 *        dimensions values
 */
inline auto get_all_dimensions() {
  static const auto dimensions = integer_pack<0, 1, 2, 3>::generate_unnamed();
  return dimensions;
}

/**
 * @brief Factory function for getting type_pack with target values
 */
inline auto get_targets() {
  static const auto targets =
      value_pack<sycl::target, sycl::target::device,
                 sycl::target::host_task>::generate_named();
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
#endif  // SYCL_CTS_COMPILING_WITH_HIPSYCL

template <accessor_type AccType>
struct tag_factory {
  static_assert(AccType != AccType,
                "There is no tag support for such accessor type");
};

/**
 * @brief Function helps to get TagT corresponding to AccessMode and Target
 * template parameters
 */
template <>
struct tag_factory<accessor_type::generic_accessor> {
  template <sycl::access_mode AccessMode, sycl::target Target>
  inline static auto get_tag() {
    if constexpr (Target == sycl::target::device) {
      if constexpr (AccessMode == sycl::access_mode::read) {
        return sycl::read_only;
      } else if constexpr (AccessMode == sycl::access_mode::write) {
        return sycl::write_only;
      } else if constexpr (AccessMode == sycl::access_mode::read_write) {
        return sycl::read_write;
      } else {
        static_assert(AccessMode != AccessMode,
                      "Unsupported sycl::access_mode");
      }
    } else if constexpr (Target == sycl::target::host_task) {
      if constexpr (AccessMode == sycl::access_mode::read) {
        return sycl::read_only_host_task;
      } else if constexpr (AccessMode == sycl::access_mode::write) {
        return sycl::write_only_host_task;
      } else if constexpr (AccessMode == sycl::access_mode::read_write) {
        return sycl::read_write_host_task;
      } else {
        static_assert(AccessMode != AccessMode,
                      "Unsupported sycl::access_mode");
      }
    } else {
      static_assert(AccessMode != AccessMode, "Unsupported sycl::target");
    }
  }
};

/**
 * @brief Function helps to get TagT corresponding to AccessMode parameter
 */
template <>
struct tag_factory<accessor_type::host_accessor> {
  template <sycl::access_mode AccessMode>
  inline static auto get_tag() {
    if constexpr (AccessMode == sycl::access_mode::read) {
      return sycl::read_only;
    } else if constexpr (AccessMode == sycl::access_mode::write) {
      return sycl::write_only;
    } else if constexpr (AccessMode == sycl::access_mode::read_write) {
      return sycl::read_write;
    } else {
      static_assert(AccessMode != AccessMode, "Unsupported sycl::access_mode");
    }
  }
};

/**
 * @brief Common function that check constructor post-conditions for empty
 * accessor, and store result in res_acc
 *
 * @tparam TestingAccT Type of testing accessor
 * @tparam ResultAccT  Type of result accessor
 * @param testing_acc Instance of TestingAccT that were constructed with default
 * constructor
 * @param res_acc Instance of result accessor
 */
template <typename TestingAccT, typename ResultAccT>
void check_empty_accessor_constructor_post_conditions(TestingAccT testing_acc,
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
// FIXME: re-enable when handler.host_task and sycl::errc is implemented in
// hipsycl and computcpp
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL && !SYCL_CTS_COMPILING_WITH_COMPUTECPP

/**
 * @brief Common function that constructs accessor with default constructor
 *and checks post-conditions
 *
 * @tparam AccType Type of the accessor
 * @tparam DataT Type of underlying data
 * @tparam Dimension Dimensions of the accessor
 * @tparam AccessMode Access mode of the accessor
 * @tparam Target Target of accessor
 * @tparam GetAccFunctorT Type of functor for accessor creation
 */
template <accessor_type AccType, typename DataT, int Dimension,
          sycl::access_mode AccessMode = sycl::access_mode::read_write,
          sycl::target Target = sycl::target::device, typename GetAccFunctorT>
void check_def_constructor(GetAccFunctorT get_accessor_functor) {
  auto queue = once_per_unit::get_queue();
  sycl::range<1> r(1);
  const size_t conditions_checks_size = 8;
  bool conditions_check[conditions_checks_size]{false};

  if constexpr (AccType != accessor_type::host_accessor) {
    sycl::buffer res_buf(conditions_check, sycl::range(conditions_checks_size));

    queue
        .submit([&](sycl::handler& cgh) {
          sycl::accessor res_acc(res_buf, cgh);
          auto acc = get_accessor_functor();
          if (acc.is_placeholder()) {
            cgh.require(acc);
          }
          if constexpr (Target == sycl::target::host_task) {
            cgh.host_task([=] {
              check_empty_accessor_constructor_post_conditions(acc, res_acc);
            });
          } else if constexpr (Target == sycl::target::device) {
            cgh.parallel_for_work_group(r, [=](sycl::group<1>) {
              check_empty_accessor_constructor_post_conditions(acc, res_acc);
            });
          }
        })
        .wait_and_throw();
  } else {
    auto acc = get_accessor_functor();
    check_empty_accessor_constructor_post_conditions(acc, conditions_check);
  }

  for (size_t i = 0; i < conditions_checks_size; i++) {
    CHECK(conditions_check[i]);
  }
}

/**
 * @brief Common function that constructs accessor with zero-length buffer
 *and checks post-conditions
 *
 * @tparam AccType Type of the accessor
 * @tparam DataT Type of underlying data
 * @tparam Dimension Dimensions of the accessor
 * @tparam AccessMode Access mode of the accessor
 * @tparam Target Target of accessor
 * @tparam GetAccFunctorT Type of functor for accessor creation
 */
template <accessor_type AccType, typename DataT, int Dimension,
          sycl::access_mode AccessMode = sycl::access_mode::read_write,
          sycl::target Target = sycl::target::device, typename GetAccFunctorT>
void check_zero_length_buffer_constructor(GetAccFunctorT get_accessor_functor) {
  auto queue = once_per_unit::get_queue();
  sycl::range<Dimension> r =
      util::get_cts_object::range<Dimension>::get(0, 0, 0);
  sycl::buffer<DataT, Dimension> data_buf(r);
  const size_t conditions_checks_size = 8;
  bool conditions_check[conditions_checks_size]{false};

  if constexpr (AccType != accessor_type::host_accessor) {
    sycl::buffer res_buf(conditions_check, sycl::range(conditions_checks_size));

    queue
        .submit([&](sycl::handler& cgh) {
          sycl::accessor res_acc(res_buf);
          auto acc = get_accessor_functor(data_buf, cgh);
          if constexpr (Target == sycl::target::host_task) {
            cgh.host_task([=] {
              check_empty_accessor_constructor_post_conditions(acc, res_acc);
            });
          } else if constexpr (Target == sycl::target::device) {
            cgh.parallel_for_work_group(r, [=](sycl::group<Dimension>) {
              check_empty_accessor_constructor_post_conditions(acc, res_acc);
            });
          }
        })
        .wait_and_throw();
  } else {
    auto acc = get_accessor_functor(data_buf);
    check_empty_accessor_constructor_post_conditions(acc, conditions_check);
  }

  for (size_t i = 0; i < conditions_checks_size; i++) {
    CHECK(conditions_check[i]);
  }
}
#endif  // !SYCL_CTS_COMPILING_WITH_HIPSYCL &&
        // !SYCL_CTS_COMPILING_WITH_COMPUTECPP

namespace detail {
/**
 * @brief Wraps callable to make possible chaining foo(boo(arg)) calls by fold
 *        expression
 */
template <typename InvocableT>
class invoke_helper {
  const InvocableT& m_action;

 public:
  invoke_helper(const InvocableT& action) : m_action(action) {}

  template <typename... ArgsT>
  decltype(auto) operator=(ArgsT&&... args) {
    static_assert(std::is_invocable_v<InvocableT, ArgsT...>, "Invalid usage");
    // Returns either by-reference or by-value depending on action return type
    return m_action(std::forward<ArgsT>(args)...);
  }
};
}  // namespace detail

/**
 * @brief Function that tries to read or/and write depending on AccessMode
 * parameter. Results of compare will be stored in res_acc
 *
 * @tparam DataT Type of underlying data
 * @tparam AccessMode Access mode of the accessor
 * @tparam AccT Type of testing accessor
 * @tparam ResultAccT Type of accessor for storing result
 * @param testing_acc Instance of sycl::accessor to read/write
 * @param res_acc Accessor for storing result
 */
template <typename DataT, sycl::access_mode AccessMode, typename AccT,
          typename ResultAccT>
void read_write_zero_dim_acc(AccT testing_acc, ResultAccT res_acc) {
  DataT other_data = value_operations::init<DataT>(expected_val);

  if constexpr (AccessMode != sycl::access_mode::write) {
    DataT acc_ref(testing_acc);
    res_acc[0] = value_operations::are_equal(acc_ref, other_data);
  }
  if constexpr (AccessMode != sycl::access_mode::read) {
    DataT& acc_ref = testing_acc;
    value_operations::assign(acc_ref, changed_val);
  }
}
// FIXME: re-enable when handler.host_task and sycl::errc is implemented in
// hipsycl and computecpp
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL && !SYCL_CTS_COMPILING_WITH_COMPUTECPP

/**
 * @brief Function helps to check zero dimension constructor of accessor
 *
 * @tparam AccType Type of the accessor
 * @tparam DataT Type of underlying data
 * @tparam Dimension Dimensions of the accessor
 * @tparam AccessMode Access mode of the accessor
 * @tparam Target Target of accessor
 * @param get_accessor_functor Functor for accessor creation
 * @param modify_accessor Functors to check either accessor modification or
 *        copy, move or conversion between accessor types; a sequence is empty
 *        by default
 */
template <accessor_type AccType, typename DataT, sycl::access_mode AccessMode,
          sycl::target Target = sycl::target::device, typename GetAccFunctorT,
          typename... ModifyAccFunctorsT>
void check_zero_dim_constructor(GetAccFunctorT get_accessor_functor,
                                ModifyAccFunctorsT... modify_accessor) {
  auto queue = once_per_unit::get_queue();
  sycl::range<1> r(1);
  DataT some_data = value_operations::init<DataT>(expected_val);

  bool compare_res = false;

  if constexpr (AccType != accessor_type::host_accessor) {
    sycl::buffer res_buf(&compare_res, r);
    sycl::buffer<DataT, 1> data_buf(&some_data, r);

    queue
        .submit([&](sycl::handler& cgh) {
          sycl::accessor res_acc(res_buf, cgh);

          auto acc = get_accessor_functor(data_buf, cgh);
          if constexpr (AccType == accessor_type::generic_accessor) {
            if (acc.is_placeholder()) {
              cgh.require(acc);
            }
          }
          if constexpr (Target == sycl::target::host_task) {
            cgh.host_task([=] {
              // We are free either to create new accessor instance or to
              // modify original accessor and provide reference to it;
              // a reference to original accessor would be used if there was
              // no any modify_accessor functor passed
              auto&& acc_instance =
                  (detail::invoke_helper{modify_accessor} = ... = acc);
              read_write_zero_dim_acc<DataT, AccessMode>(acc_instance, res_acc);
            });
          } else if constexpr (Target == sycl::target::device) {
            cgh.parallel_for_work_group(r, [=](sycl::group<1>) {
              auto&& acc_instance =
                  (detail::invoke_helper{modify_accessor} = ... = acc);
              read_write_zero_dim_acc<DataT, AccessMode>(acc_instance, res_acc);
            });
          } else {
            static_assert(Target != Target, "Unexpected accessor type");
          }
        })
        .wait_and_throw();
  } else {
    sycl::buffer<DataT, 1> data_buf(&some_data, r);
    auto acc = get_accessor_functor(data_buf);
    auto&& acc_instance = (detail::invoke_helper{modify_accessor} = ... = acc);

    // Argument for storing result should support subscript operator
    bool compare_res_arr[1]{false};
    read_write_zero_dim_acc<DataT, AccessMode>(acc_instance, compare_res_arr);
    compare_res = compare_res_arr[0];
  }

  if constexpr (AccessMode != sycl::access_mode::write) {
    CHECK(compare_res);
  }

  // When testing local_accessor we should skip this check, as local
  // accessor can't modify host memory
  if constexpr (AccType != accessor_type::local_accessor) {
    if constexpr (AccessMode != sycl::access_mode::read) {
      CHECK(value_operations::are_equal(some_data, changed_val));
    }
  }
}
#endif  // !SYCL_CTS_COMPILING_WITH_HIPSYCL &&
        // !SYCL_CTS_COMPILING_WITH_COMPUTECPP

/**
 * @brief Function that tries to read or/and write depending on AccessMode
 * parameter. Results of compare will be stored in res_acc
 *
 * @tparam DataT Type of underlying data
 * @tparam Dimension Dimensions of the accessor
 * @tparam AccessMode Access mode of the accessor
 * @tparam AccT Type of testing accessor
 * @tparam ResultAccT Type of accessor for storing result
 * @param testing_acc Instance of sycl::accessor to read/write
 * @param res_acc Accessor for storing result
 */
template <typename DataT, int Dimension, sycl::access_mode AccessMode,
          typename AccT, typename ResultAccT>
void read_write_acc(AccT testing_acc, ResultAccT res_acc) {
  DataT other_data = value_operations::init<DataT>(expected_val);
  auto id = util::get_cts_object::id<Dimension>::get(0, 0, 0);

  if constexpr (AccessMode != sycl::access_mode::write) {
    res_acc[0] = value_operations::are_equal(testing_acc[id], other_data);
  }
  if constexpr (AccessMode != sycl::access_mode::read) {
    value_operations::assign(testing_acc[id], changed_val);
  }
}

// FIXME: re-enable when handler.host_task and sycl::errc is implemented in
// hipsycl and computecpp
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL && !SYCL_CTS_COMPILING_WITH_COMPUTECPP

/**
 * @brief Function helps to check common constructor of accessor
 *
 * @tparam AccType Type of the accessor
 * @tparam DataT Type of underlying data
 * @tparam Dimension Dimensions of the accessor
 * @tparam AccessMode Access mode of the accessor
 * @tparam Target Target of accessor
 * @param r Range for accessors buffer
 * @param get_accessor_functor Functor for accessor creation
 * @param modify_accessor Functors to check either accesor modification or copy,
 *         move or conversion between accessor types; a sequence is empty by
 *         default
 */
template <accessor_type AccType, typename DataT, int Dimension,
          sycl::access_mode AccessMode,
          sycl::target Target = sycl::target::device, typename GetAccFunctorT,
          typename... ModifyAccFunctorsT>
void check_common_constructor(const sycl::range<Dimension>& r,
                              GetAccFunctorT get_accessor_functor,
                              ModifyAccFunctorsT... modify_accessor) {
  auto queue = once_per_unit::get_queue();
  bool compare_res = false;
  DataT some_data = value_operations::init<DataT>(expected_val);

  if constexpr (AccType != accessor_type::host_accessor) {
    sycl::buffer res_buf(&compare_res, sycl::range(1));
    sycl::buffer<DataT, Dimension> data_buf(&some_data, r);

    queue
        .submit([&](sycl::handler& cgh) {
          sycl::accessor res_acc(res_buf);
          auto acc = get_accessor_functor(data_buf, cgh);

          if constexpr (AccType == accessor_type::generic_accessor) {
            if (acc.is_placeholder()) {
              cgh.require(acc);
            }
          }

          if constexpr (Target == sycl::target::host_task) {
            cgh.host_task([=] {
              auto&& acc_instance =
                  (detail::invoke_helper{modify_accessor} = ... = acc);
              read_write_acc<DataT, Dimension, AccessMode>(acc_instance,
                                                           res_acc);
            });
          } else if constexpr (Target == sycl::target::device) {
            cgh.parallel_for_work_group(
                sycl::range(1), [=](sycl::group<1>) {
                  auto&& acc_instance =
                      (detail::invoke_helper{modify_accessor} = ... = acc);
                  read_write_acc<DataT, Dimension, AccessMode>(acc_instance,
                                                               res_acc);
                });
          } else {
            static_assert(Target != Target, "Unexpected accessor type");
          }
        })
        .wait_and_throw();
  } else {
    sycl::buffer<DataT, Dimension> data_buf(&some_data, r);
    auto acc = get_accessor_functor(data_buf);
    auto&& acc_instance = (detail::invoke_helper{modify_accessor} = ... = acc);

    // Argument for storing result should support subscript operator
    bool compare_res_arr[1]{false};
    read_write_acc<DataT, Dimension, AccessMode>(acc_instance, compare_res_arr);
    compare_res = compare_res_arr[0];
  }

  if constexpr (AccessMode != sycl::access_mode::write) {
    CHECK(compare_res);
  }

  // When testing local_accessor we should skip this check, as local
  // accessor can't modify host memory
  if constexpr (AccType != accessor_type::local_accessor) {
    if constexpr (AccessMode != sycl::access_mode::read) {
      CHECK(value_operations::are_equal(some_data, changed_val));
    }
  }
}

/**
 * @brief Function helps to check if passing of a placeholder accessor triggers
 * the exception
 *
 * @tparam AccType Type of the accessor
 * @tparam DataT Type of underlying data
 * @tparam Dimension Dimensions of the accessor
 * @tparam AccessMode Access mode of the accessor
 * @tparam Target Target of accessor
 * @param r Range for accessors buffer
 * @param get_accessor_functor Functor for accessor creation
 */
template <typename DataT, int Dimension, typename GetAccFunctorT, typename OpT>
void run_placeholder_accessor_exception(const sycl::range<Dimension>& r,
                                        GetAccFunctorT get_accessor_functor,
                                        OpT memory_operation,
                                        std::string op_name) {
  auto queue = once_per_unit::get_queue();
  DataT some_data = value_operations::init<DataT>(expected_val);
  using T = std::remove_const_t<DataT>;
  T other_data = value_operations::init<T>(changed_val);
  {
    sycl::buffer<DataT, Dimension> data_buf(&some_data, r);
    sycl::buffer<T, Dimension> other_buf(&other_data, r);

    auto action = [&] {
      queue
          .submit([&](sycl::handler& cgh) {
            auto acc = get_accessor_functor(data_buf);
            sycl::accessor<T, Dimension> other_acc(other_buf, cgh);
            memory_operation(acc, cgh, other_acc);
          })
          .wait_and_throw();
    };
    INFO(
        "Implementation has to throw a sycl::exception with "
        "sycl::errc::kernel_argument when a placeholder accessor is passed to "
        "the command. Check for " +
        op_name);
    CHECK_THROWS_MATCHES(
        action(), sycl::exception,
        sycl_cts::util::equals_exception(sycl::errc::kernel_argument));
  }
}

/**
 * @brief Function helps to check if passing of a placeholder accessor triggers
 * the exception
 *
 * @tparam DataT Type of underlying data
 * @tparam Dimension Dimensions of the accessor
 * @tparam AccessMode Access mode of the accessor
 * @tparam Target Target of accessor
 * @param r Range for accessors buffer
 * @param get_accessor_functor Functor for accessor creation
 */
template <typename DataT, int Dimension, sycl::access_mode AccessMode,
          sycl::target Target, typename GetAccFunctorT>
void check_placeholder_accessor_exception(const sycl::range<Dimension>& r,
                                          GetAccFunctorT get_accessor_functor) {
  if constexpr (Target == sycl::target::host_task) {
    auto host_task = [&](auto& acc, auto& cgh, auto& acc2) {
      cgh.host_task([=] { auto copy_acc = acc; });
    };
    run_placeholder_accessor_exception<DataT>(r, get_accessor_functor,
                                              host_task, "host_task");
  } else {
    auto single_task = [&](auto& acc, auto& cgh, auto& acc2) {
      cgh.single_task([=] { auto copy_acc = acc; });
    };
    run_placeholder_accessor_exception<DataT>(r, get_accessor_functor,
                                              single_task, "single_task");

    auto parallel_for_range = [&](auto& acc, auto& cgh, auto& acc2) {
      cgh.parallel_for(r, [=](auto item) { auto copy_acc = acc; });
    };
    run_placeholder_accessor_exception<DataT>(
        r, get_accessor_functor, parallel_for_range, "parallel_for with range");

    auto parallel_for_nd_range = [&](auto& acc, auto& cgh, auto& acc2) {
      cgh.parallel_for(sycl::nd_range<Dimension>(r, r),
                       [=](auto nd_item) { auto copy_acc = acc; });
    };
    run_placeholder_accessor_exception<DataT>(r, get_accessor_functor,
                                              parallel_for_nd_range,
                                              "parallel_for with nd_range");

    auto parallel_for_work_group = [&](auto& acc, auto& cgh, auto& acc2) {
      cgh.parallel_for_work_group(r, [=](auto group) { auto copy_acc = acc; });
    };
    run_placeholder_accessor_exception<DataT>(r, get_accessor_functor,
                                              parallel_for_work_group,
                                              "parallel_for_work_group");

    auto update_host = [&](auto& acc, auto& cgh, auto& acc2) {
      cgh.update_host(acc);
    };
    run_placeholder_accessor_exception<DataT>(r, get_accessor_functor,
                                              update_host, "update_host");

    if constexpr (AccessMode != sycl::access_mode::write) {
      auto copy_to_shared = [&](auto& acc, auto& cgh, auto& acc2) {
        using T = std::remove_const_t<DataT>;
        std::shared_ptr<T> dest(new T);
        cgh.copy(acc, dest);
      };
      run_placeholder_accessor_exception<DataT>(
          r, get_accessor_functor, copy_to_shared, "copy to std::shared_ptr");

      auto copy_to_pointer = [&](auto& acc, auto& cgh, auto& acc2) {
        using T = std::remove_const_t<DataT>;
        T val;
        T* dest = &val;
        cgh.copy(acc, dest);
      };
      run_placeholder_accessor_exception<DataT>(
          r, get_accessor_functor, copy_to_pointer, "copy to pointer");

      auto copy_to_acc = [&](auto& acc, auto& cgh, auto& acc2) {
        cgh.copy(acc, acc2);
      };
      run_placeholder_accessor_exception<DataT>(
          r, get_accessor_functor, copy_to_acc, "copy to new accessor");
    }

    if constexpr (AccessMode != sycl::access_mode::read) {
      auto copy_shared = [&](auto& acc, auto& cgh, auto& acc2) {
        using T = std::remove_const_t<DataT>;
        std::shared_ptr<T> src(new T);
        cgh.copy(src, acc);
      };
      run_placeholder_accessor_exception<DataT>(
          r, get_accessor_functor, copy_shared, "copy std::shared_ptr");

      auto copy_pointer = [&](auto& acc, auto& cgh, auto& acc2) {
        using T = std::remove_const_t<DataT>;
        T val;
        T* src = &val;
        cgh.copy(src, acc);
      };
      run_placeholder_accessor_exception<DataT>(r, get_accessor_functor,
                                                copy_shared, "copy pointer");

      auto copy_acc = [&](auto& acc, auto& cgh, auto& acc2) {
        cgh.copy(acc2, acc);
      };
      run_placeholder_accessor_exception<DataT>(r, get_accessor_functor,
                                                copy_acc, "copy new accessor");

      auto fill = [&](auto& acc, auto& cgh, auto& acc2) {
        using T = std::remove_const_t<DataT>;
        T val = T(changed_val);
        cgh.fill(acc, val);
      };
      run_placeholder_accessor_exception<DataT>(r, get_accessor_functor, fill,
                                                "fill");
    }
  }
}
#endif  // !SYCL_CTS_COMPILING_WITH_HIPSYCL &&
        // !SYCL_CTS_COMPILING_WITH_COMPUTECPP
/**
 * @brief Function mainly for testing no_init property. The function tries to
 * write to the accessor and only after that tries to read from the accessor.
 *
 * @tparam AccT Type of testing accessor
 * @tparam ResultAccT Type of accessor for storing result
 * @param testing_acc Instance of sycl::accessor to read/write
 * @param res_acc Accessor for storing result
 */
template <typename DataT, int Dimension, sycl::access_mode AccessMode,
          typename AccT, typename ResultAccT>
void write_read_acc(AccT testing_acc, ResultAccT res_acc) {
  DataT expected_data = value_operations::init<DataT>(changed_val);
  auto id = util::get_cts_object::id<Dimension>::get(0, 0, 0);

  value_operations::assign(testing_acc[id], changed_val);

  if constexpr (AccessMode == sycl::access_mode::read_write) {
    res_acc[0] = value_operations::are_equal(testing_acc[id], expected_data);
  }
}
// FIXME: re-enable when handler.host_task and sycl::errc is implemented
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL && !SYCL_CTS_COMPILING_WITH_COMPUTECPP

template <accessor_type AccType, typename DataT, int Dimension,
          sycl::access_mode AccessMode, sycl::target Target>
class kernel_no_init_prop;

/**
 * @brief Function helps to check accessor constructor with no_init property
 *
 * @tparam GetAccFunctorT Type of functor that constructs testing accessor
 */
template <accessor_type AccType, typename DataT, int Dimension,
          sycl::access_mode AccessMode,
          sycl::target Target = sycl::target::device, typename GetAccFunctorT>
void check_no_init_prop(GetAccFunctorT get_accessor_functor,
                        const sycl::range<Dimension> r) {
  auto queue = once_per_unit::get_queue();
  bool compare_res = false;
  DataT some_data = value_operations::init<DataT>(expected_val);

  if constexpr (AccType != accessor_type::host_accessor) {
    sycl::buffer res_buf(&compare_res, sycl::range(1));
    sycl::buffer<DataT, Dimension> data_buf(&some_data, r);

    queue
        .submit([&](sycl::handler& cgh) {
          sycl::accessor res_acc(res_buf, cgh);

          auto acc = get_accessor_functor(data_buf, cgh);

          if (Target == sycl::target::host_task) {
            cgh.host_task([=] {
              write_read_acc<DataT, Dimension, AccessMode>(acc, res_acc);
            });
          } else if (Target == sycl::target::device) {
            cgh.parallel_for_work_group(sycl::range(1), [=](sycl::group<1>) {
              write_read_acc<DataT, Dimension, AccessMode>(acc, res_acc);
            });
          }
        })
        .wait_and_throw();
  } else {
    sycl::buffer<DataT, Dimension> data_buf(&some_data, r);
    auto acc = get_accessor_functor(data_buf);
    // Argument for storing result should support subscript operator
    bool compare_res_arr[1]{false};
    write_read_acc<DataT, Dimension, AccessMode>(acc, compare_res_arr);
    compare_res = compare_res_arr[0];
  }

  CHECK(value_operations::are_equal(some_data, changed_val));
  if constexpr (AccessMode == sycl::access_mode::read_write) {
    CHECK(compare_res);
  }
}

/**
 * @brief Function helps to verify that constructor of accessor with no_init
 * property and access_mode::read triggers an exception
 *
 * @tparam GetAccFunctorT Type of functor that constructs testing accessor
 */
template <accessor_type AccType, typename DataT, int Dimension,
          sycl::target Target = sycl::target::device, typename GetAccFunctorT>
void check_no_init_prop_exception(GetAccFunctorT construct_acc,
                                  const sycl::range<Dimension> r) {
  auto queue = once_per_unit::get_queue();
  DataT some_data = value_operations::init<DataT>(expected_val);
  {
    sycl::buffer<DataT, Dimension> data_buf(&some_data, r);

    if constexpr (AccType != accessor_type::host_accessor) {
      auto action = [&] { construct_acc(queue, data_buf); };
      CHECK_THROWS_MATCHES(
          action(), sycl::exception,
          sycl_cts::util::equals_exception(sycl::errc::invalid));
    } else {
      auto action = [&] { construct_acc(data_buf); };
      CHECK_THROWS_MATCHES(
          action(), sycl::exception,
          sycl_cts::util::equals_exception(sycl::errc::invalid));
    }
  }
}
#endif  // !SYCL_CTS_COMPILING_WITH_HIPSYCL &&
        // !SYCL_CTS_COMPILING_WITH_COMPUTECPP
/**
 * @brief Function tests AccT::get_pointer() method

 * @tparam AccT Type of testing accessor
 * @tparam T Type of underlying data
 */
template <typename T, typename AccT>
void test_accessor_ptr(AccT& accessor, T expected_data) {
  INFO("check get_pointer() method");
  auto acc_pointer = accessor.get_pointer();
  STATIC_CHECK(std::is_same_v<decltype(acc_pointer),
                              std::add_pointer_t<typename AccT::value_type>>);
  CHECK(value_operations::are_equal(*acc_pointer, expected_data));
}

// FIXME: re-enable when sycl::access::decorated enumeration is implemented in
// hipsycl and computecpp
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL && !SYCL_CTS_COMPILING_WITH_COMPUTECPP
template <typename T, typename AccT, typename AccRes>
void test_accessor_ptr_device(AccT& accessor, T expected_data,
                              AccRes& res_acc) {
  auto acc_multi_ptr_no =
      accessor.template get_multi_ptr<sycl::access::decorated::no>();
  res_acc[0] = std::is_same_v<
      decltype(acc_multi_ptr_no),
      typename AccT::template accessor_ptr<sycl::access::decorated::no>>;
  res_acc[0] &=
      value_operations::are_equal(*acc_multi_ptr_no.get(), expected_data);

  auto acc_multi_ptr_yes =
      accessor.template get_multi_ptr<sycl::access::decorated::yes>();
  res_acc[0] &= std::is_same_v<
      decltype(acc_multi_ptr_yes),
      typename AccT::template accessor_ptr<sycl::access::decorated::yes>>;
  res_acc[0] &=
      value_operations::are_equal(*acc_multi_ptr_yes.get(), expected_data);

  auto acc_pointer = accessor.get_pointer();
  res_acc[0] &= std::is_same_v<decltype(acc_pointer),
                               std::add_pointer_t<typename AccT::value_type>>;
  res_acc[0] &= value_operations::are_equal(*acc_pointer, expected_data);
}
#endif  // !SYCL_CTS_COMPILING_WITH_HIPSYCL &&
        // !SYCL_CTS_COMPILING_WITH_COMPUTECPP
/**
 * @brief Function checks common buffer and local accessor member functions
 */
template <typename AccT>
void test_accessor_methods_common(const AccT& accessor,
                                  const size_t expected_byte_size,
                                  const size_t expected_size) {
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
}

template <typename AccT, int dims>
void test_accessor_get_range_method(const AccT& accessor,
                                    const sycl::range<dims>& expected_range) {
  {
    INFO("check get_range() method");
    auto acc_range = accessor.get_range();
    STATIC_CHECK(std::is_same_v<decltype(acc_range), sycl::range<dims>>);
    CHECK(acc_range == expected_range);
  }
}

template <typename AccT, int dims>
void test_accessor_get_offset_method(const AccT& accessor,
                                     const sycl::id<dims>& expected_offset) {
  INFO("check get_offset() method");
  auto acc_offset = accessor.get_offset();
  STATIC_CHECK(std::is_same_v<decltype(acc_offset), sycl::id<dims>>);
  CHECK(acc_offset == expected_offset);
}

template <typename AccT, int dims>
void test_accessor_range_methods(const AccT& accessor,
                                 const sycl::range<dims>& expected_range,
                                 const sycl::id<dims>& expected_offset) {
  test_accessor_get_range_method<AccT, dims>(accessor, expected_range);
  test_accessor_get_offset_method<AccT, dims>(accessor, expected_offset);
}

/**
 * @brief Function invokes \c has_property() member function with \c PropT
 * property and verifies that true returns
 *
 * @tparam GetAccFunctorT Type of functor that constructs testing accessor
 */
template <accessor_type AccType, typename DataT, int Dimension, typename PropT,
          typename GetAccFunctorT>
void check_has_property_member_func(GetAccFunctorT construct_acc,
                                    const sycl::range<Dimension> r) {
  auto queue = once_per_unit::get_queue();
  bool compare_res = false;
  DataT some_data = value_operations::init<DataT>(expected_val);
  sycl::buffer<DataT, Dimension> data_buf(&some_data, r);

  if constexpr (AccType != accessor_type::host_accessor) {
    queue
        .submit([&](sycl::handler& cgh) {
          auto acc = construct_acc(data_buf, cgh);
          compare_res = acc.template has_property<sycl::property::no_init>();
        })
        .wait_and_throw();
  } else {
    auto acc = construct_acc(data_buf);
    compare_res = acc.template has_property<sycl::property::no_init>();
  }
  CHECK(compare_res);
}

/**
 * @brief Function invokes \c has_property() member function without \c PropT
 * property and verifies that false returns
 *
 * @tparam GetAccFunctorT Type of functor that constructs testing accessor
 */
template <accessor_type AccType, typename DataT, int Dimension,
          typename GetAccFunctorT>
void check_has_property_member_without_no_init(GetAccFunctorT construct_acc,
                                               const sycl::range<Dimension> r) {
  auto queue = once_per_unit::get_queue();
  bool compare_res = false;
  DataT some_data = value_operations::init<DataT>(expected_val);
  sycl::buffer<DataT, Dimension> data_buf(&some_data, r);

  if constexpr (AccType != accessor_type::host_accessor) {
    queue
        .submit([&](sycl::handler& cgh) {
          auto acc = construct_acc(data_buf, cgh);
          compare_res = acc.template has_property<sycl::property::no_init>();
        })
        .wait_and_throw();
  } else {
    auto acc = construct_acc(data_buf);
    compare_res = acc.template has_property<sycl::property::no_init>();
  }
  CHECK(!compare_res);
}

/**
 * @brief Function invokes \c get_property() member function with \c PropT
 * property and verifies that exception occures
 *
 * @tparam GetAccFunctorT Type of functor that constructs testing accessor
 */
template <accessor_type AccType, typename DataT, int Dimension, typename PropT,
          typename GetAccFunctorT>
void check_get_property_member_func(GetAccFunctorT construct_acc,
                                    const sycl::range<Dimension> r) {
  auto queue = once_per_unit::get_queue();
  DataT some_data = value_operations::init<DataT>(expected_val);
  sycl::buffer<DataT, Dimension> data_buf(&some_data, r);

  if constexpr (AccType != accessor_type::host_accessor) {
    queue
        .submit([&](sycl::handler& cgh) {
          auto acc = construct_acc(data_buf, cgh);
          auto acc_prop = acc.template get_property<PropT>();
          CHECK(std::is_same_v<PropT, decltype(acc_prop)>);
        })
        .wait_and_throw();
  } else {
    auto acc = construct_acc(data_buf);
    auto acc_prop = acc.template get_property<PropT>();
    CHECK(std::is_same_v<PropT, decltype(acc_prop)>);
  }
}

// FIXME: re-enable when handler.host_task and sycl::errc is implemented in
// hipsycl and computecpp
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL && !SYCL_CTS_COMPILING_WITH_COMPUTECPP
/**
 * @brief Function invokes \c get_property() member function without \c PropT
 * property and verifies that false returns
 *
 * @tparam GetAccFunctorT Type of functor that constructs testing accessor
 */
template <accessor_type AccType, typename DataT, int Dimension,
          typename GetAccFunctorT>
void check_get_property_member_without_no_init(GetAccFunctorT construct_acc,
                                               const sycl::range<Dimension> r) {
  auto queue = once_per_unit::get_queue();
  DataT some_data = value_operations::init<DataT>(expected_val);
  sycl::buffer<DataT, Dimension> data_buf(&some_data, r);

  if constexpr (AccType != accessor_type::host_accessor) {
    queue
        .submit([&](sycl::handler& cgh) {
          auto acc = construct_acc(data_buf, cgh);
          auto action = [&] {
            acc.template get_property<sycl::property::no_init>();
          };
          CHECK_THROWS_MATCHES(
              action(), sycl::exception,
              sycl_cts::util::equals_exception(sycl::errc::invalid));
        })
        .wait_and_throw();
  } else {
    auto acc = construct_acc(data_buf);
    auto action = [&] { acc.template get_property<sycl::property::no_init>(); };
    CHECK_THROWS_MATCHES(action(), sycl::exception,
                         sycl_cts::util::equals_exception(sycl::errc::invalid));
  }
}
#endif  // SYCL_CTS_COMPILING_WITH_HIPSYCL &&
        // !SYCL_CTS_COMPILING_WITH_COMPUTECPP
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
decltype(auto) get_subscript_overload(const AccT& accessor, size_t index) {
  if constexpr (dims == 1) return accessor[index];
  if constexpr (dims == 2) return accessor[index][index];
  if constexpr (dims == 3) return accessor[index][index][index];
}

// Helper function to increment id linearly according to its linear position
// id2+(id1*r2)+(id0*r1*r2) for dims = 3 or id1+(id0*r1) for dims = 2
//
// Incrementing index of last dimension if it not yet reached end of the range
// otherwise reseting it and trying to increment index of previous dimension.
template <int dims>
sycl::id<dims> next_id_linearly(sycl::id<dims> id, size_t size) {
  for (int i = dims - 1; i >= 0; i--) {
    if (id[i] < size - 1) {
      id[i]++;
      break;
    } else {
      id[i] = 0;
    }
  }
  return id;
}

// FIXME: re-enable when handler.host_task is implemented in hipsycl
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL

template <accessor_type AccType, typename T, int dims,
          sycl::access_mode AccessMode, sycl::target Target>
class kernel_linearization;

/**
 * @brief Common function that checks correct linearization for generic
 *        and host constructors
 *
 * @tparam AccType Type of the accessor
 * @tparam T Type of underlying data
 * @tparam Dimension Dimensions of the accessor
 * @tparam AccessMode Access mode of the accessor
 * @tparam Target Target of accessor
 */
template <accessor_type AccType, typename T, int dims,
          sycl::access_mode AccessMode = sycl::access_mode::read_write,
          sycl::target Target = sycl::target::device>
void check_linearization() {
  constexpr size_t range_size = 2;
  constexpr size_t buff_size = (dims == 3) ? 2 * 2 * 2 : 2 * 2;

  auto range = util::get_cts_object::range<dims>::get(range_size, range_size,
                                                      range_size);

  std::remove_const_t<T> data[buff_size];
  std::iota(data, (data + range.size()), 0);
  sycl::buffer<T, dims> data_buf(data, range);

  if constexpr (AccType != accessor_type::host_accessor) {
    auto queue = once_per_unit::get_queue();
    auto r = util::get_cts_object::range<dims>::get(1, 1, 1);
    bool res = true;
    {
      sycl::buffer<T, dims> data_buf(data, range);
      sycl::buffer res_buf(&res, sycl::range(1));
      queue
          .submit([&](sycl::handler& cgh) {
            sycl::accessor<T, dims, AccessMode, Target> acc(data_buf, cgh);

            if constexpr (Target == sycl::target::device) {
              sycl::accessor res_acc(res_buf, cgh);
              cgh.single_task([=] {
                sycl::id<dims> id{};
                for (auto& elem : acc) {
                  res_acc[0] &= value_operations::are_equal(elem, acc[id]);
                  id = next_id_linearly(id, range_size);
                }
              });
            } else {
              cgh.host_task([=] {
                sycl::id<dims> id{};
                for (auto& elem : acc) {
                  CHECK(value_operations::are_equal(elem, acc[id]));
                  id = next_id_linearly(id, range_size);
                }
              });
            }
          })
          .wait_and_throw();
    }
    if constexpr (Target == sycl::target::device) CHECK(res);

  } else {
    sycl::host_accessor<T, dims, AccessMode> acc(data_buf);
    sycl::id<dims> id{};
    for (auto& elem : acc) {
      CHECK(value_operations::are_equal(elem, acc[id]));
      id = next_id_linearly(id, range_size);
    }
  }
}
#endif

template <int dims, typename AccT>
typename AccT::reference get_accessor_reference(const AccT& acc) {
  if constexpr (0 == dims) {
    typename AccT::reference aref = acc;
    return aref;
  } else {
    return acc[sycl::id<dims>()];
  }
}

template <typename AccT, typename T = int>
void test_begin_end_host(AccT& accessor, T exp_first = {}, T exp_last = {},
                         bool empty = true) {
  {
    INFO("check begin() method");
    auto it = accessor.begin();
    STATIC_CHECK(std::is_same_v<decltype(it), typename AccT::iterator>);
    if (!empty) CHECK(value_operations::are_equal(*it, exp_first));
  }
  {
    INFO("check end() method");
    auto it = accessor.end();
    STATIC_CHECK(std::is_same_v<decltype(it), typename AccT::iterator>);
    if (!empty) CHECK(value_operations::are_equal(*(--it), exp_last));
  }
  {
    INFO("check cbegin() method");
    auto it = accessor.cbegin();
    STATIC_CHECK(std::is_same_v<decltype(it), typename AccT::const_iterator>);
    if (!empty) CHECK(value_operations::are_equal(*it, exp_first));
  }
  {
    INFO("check cend() method");
    auto it = accessor.cend();
    STATIC_CHECK(std::is_same_v<decltype(it), typename AccT::const_iterator>);
    if (!empty) CHECK(value_operations::are_equal(*(--it), exp_last));
  }
  {
    INFO("check rbegin() method");
    auto it = accessor.rbegin();
    STATIC_CHECK(std::is_same_v<decltype(it), typename AccT::reverse_iterator>);
    if (!empty) CHECK(value_operations::are_equal(*it, exp_last));
  }
  {
    INFO("check rend() method");
    auto it = accessor.rend();
    STATIC_CHECK(std::is_same_v<decltype(it), typename AccT::reverse_iterator>);
    if (!empty) CHECK(value_operations::are_equal(*(--it), exp_first));
  }
  {
    INFO("check crbegin() method");
    auto it = accessor.crbegin();
    STATIC_CHECK(
        std::is_same_v<decltype(it), typename AccT::const_reverse_iterator>);
    if (!empty) CHECK(value_operations::are_equal(*it, exp_last));
  }
  {
    INFO("check crend() method");
    auto it = accessor.crend();
    STATIC_CHECK(
        std::is_same_v<decltype(it), typename AccT::const_reverse_iterator>);
    if (!empty) CHECK(value_operations::are_equal(*(--it), exp_first));
  }
}

template <typename AccT, typename T = int>
bool test_begin_end_device(AccT& accessor, T exp_first = {}, T exp_last = {},
                           bool empty = true) {
  auto it_begin = accessor.begin();
  bool res = std::is_same_v<decltype(it_begin), typename AccT::iterator>;
  auto it_end = accessor.end();
  res &= std::is_same_v<decltype(it_end), typename AccT::iterator>;

  auto it_cbegin = accessor.cbegin();
  res &= std::is_same_v<decltype(it_cbegin), typename AccT::const_iterator>;

  auto it_cend = accessor.cend();
  res &= std::is_same_v<decltype(it_cend), typename AccT::const_iterator>;

  auto it_rbegin = accessor.rbegin();
  res &= std::is_same_v<decltype(it_rbegin), typename AccT::reverse_iterator>;

  auto it_rend = accessor.rend();
  res &= std::is_same_v<decltype(it_rend), typename AccT::reverse_iterator>;

  auto it_crbegin = accessor.crbegin();
  res &= std::is_same_v<decltype(it_crbegin),
                        typename AccT::const_reverse_iterator>;

  auto it_crend = accessor.crend();
  res &=
      std::is_same_v<decltype(it_crend), typename AccT::const_reverse_iterator>;

  if (!empty) {
    res &= value_operations::are_equal(*it_begin, exp_first);
    res &= value_operations::are_equal(*(--it_end), exp_last);
    res &= value_operations::are_equal(*it_cbegin, exp_first);
    res &= value_operations::are_equal(*(--it_cend), exp_last);
    res &= value_operations::are_equal(*it_rbegin, exp_last);
    res &= value_operations::are_equal(*(--it_rend), exp_first);
    res &= value_operations::are_equal(*it_crbegin, exp_last);
    res &= value_operations::are_equal(*(--it_crend), exp_first);
  }

  return res;
}

}  // namespace accessor_tests_common

#endif  // SYCL_CTS_ACCESSOR_COMMON_H

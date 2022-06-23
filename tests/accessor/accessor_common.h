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
#include "../common/section_name_builder.h"
#include "../common/type_coverage.h"
#include "../common/value_operations.h"

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
#if !SYCL_CTS_ENABLE_FULL_CONFORMANCE
  return get_lightweight_type_pack();
#else
  return get_full_conformance_type_pack();
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
  auto queue = util::get_cts_object::queue();
  sycl::range<1> r(1);
  const size_t conditions_checks_size = 8;
  bool conditions_check[conditions_checks_size]{false};

  if constexpr (AccType != accessor_type::host_accessor) {
    sycl::buffer res_buf(conditions_check, sycl::range(conditions_checks_size));

    queue
        .submit([&](sycl::handler& cgh) {
          sycl::accessor res_acc(res_buf);
          auto acc = get_accessor_functor();
          if constexpr (Target == sycl::target::host_task) {
            cgh.host_task(
                [=] { check_def_constructor_post_conditions(acc, res_acc); });
          } else if constexpr (Target == sycl::target::device) {
            cgh.parallel_for_work_group(r, [acc, res_acc](sycl::group<1>) {
              check_def_constructor_post_conditions(acc, res_acc);
            });
          }
        })
        .wait_and_throw();
  } else {
    auto acc = get_accessor_functor();
    check_def_constructor_post_conditions(acc, conditions_check);
  }

  for (size_t i = 0; i < conditions_checks_size; i++) {
    CHECK(conditions_check[i]);
  }
}

namespace detail {
/**
 * @brief Wraps callable to make possible chaining foo(boo(arg)) calls by fold
 *        expression
 */
template <typename InvocableT>
class invoke_helper {
  const InvocableT& m_action;

 public:
  invoke_helper(const InvocableT& action) : m_action(action){}

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
  DataT other_data(expected_val);

  if constexpr (AccessMode != sycl::access_mode::write) {
    DataT acc_ref(testing_acc);
    res_acc[0] = value_operations::are_equal(acc_ref, other_data);
  }
  if constexpr (AccessMode != sycl::access_mode::read) {
    DataT acc_ref(testing_acc);
    value_operations::assign(acc_ref, changed_val);
  }
}

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
  auto queue = util::get_cts_object::queue();
  sycl::range<1> r(1);
  DataT some_data(expected_val);

  bool compare_res = false;

  if constexpr (AccType != accessor_type::host_accessor) {
    sycl::buffer res_buf(&compare_res, r);
    sycl::buffer<DataT, 1> data_buf(&some_data, r);

    queue
        .submit([&](sycl::handler& cgh) {
          sycl::accessor res_acc(res_buf);

          auto acc = get_accessor_functor(data_buf, cgh);
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
  DataT other_data(expected_val);
  auto id = util::get_cts_object::id<Dimension>::get(0, 0, 0);

  if constexpr (AccessMode != sycl::access_mode::write) {
    res_acc[0] = value_operations::are_equal(testing_acc[id], other_data);
  }
  if constexpr (AccessMode != sycl::access_mode::read) {
    value_operations::assign(testing_acc[id], changed_val);
  }
}

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
  auto queue = util::get_cts_object::queue();
  bool compare_res = false;
  DataT some_data(expected_val);

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
            cgh.parallel_for_work_group(sycl::range(1), [=](sycl::group<1>) {
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
template <accessor_type AccType, typename DataT, int Dimension,
          sycl::access_mode AccessMode, sycl::target Target,
          typename GetAccFunctorT>
void check_placeholder_accessor_exception(const sycl::range<Dimension>& r,
                                          GetAccFunctorT get_accessor_functor) {
  auto queue = util::get_cts_object::queue();
  DataT some_data(expected_val);
  bool is_placeholder = false;
  {
    sycl::buffer<DataT, Dimension> data_buf(&some_data, r);

    auto action = [&] {
      queue
          .submit([&](sycl::handler& cgh) {
            auto acc = get_accessor_functor(data_buf);
            is_placeholder = acc.is_placeholder();
            if constexpr (Target == sycl::target::host_task) {
              cgh.host_task([=] {});
            } else if constexpr (Target == sycl::target::device) {
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
  DataT expected_data(changed_val);
  auto id = util::get_cts_object::id<Dimension>::get(0, 0, 0);

  value_operations::assign(testing_acc[id], changed_val);

  if constexpr (AccessMode == sycl::access_mode::read_write) {
    res_acc[0] = value_operations::are_equal(testing_acc[id], expected_data);
  }
}

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
  auto queue = util::get_cts_object::queue();
  bool compare_res = false;
  DataT some_data(expected_val);

  if constexpr (AccType != accessor_type::host_accessor) {
    sycl::buffer res_buf(&compare_res, sycl::range(1));
    sycl::buffer<DataT, Dimension> data_buf(&some_data, r);

    queue
        .submit([&](sycl::handler& cgh) {
          sycl::accessor res_acc(res_buf);

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
  auto queue = util::get_cts_object::queue();
  DataT some_data(expected_val);
  {
    sycl::buffer<DataT, Dimension> data_buf(&some_data, r);

    if constexpr (AccType != accessor_type::host_accessor) {
      auto action = [&] { construct_acc(queue, data_buf); };
      CHECK_THROWS_MATCHES(
          action, sycl::exception,
          sycl_cts::util::equals_exception(sycl::errc::invalid));
    } else {
      auto action = [&] { construct_acc(data_buf); };
      CHECK_THROWS_MATCHES(
          action, sycl::exception,
          sycl_cts::util::equals_exception(sycl::errc::invalid));
    }
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
 * @brief Function invokes \c has_property() member function with \c PropT
 * property and verifies that true returns
 *
 * @tparam GetAccFunctorT Type of functor that constructs testing accessor
 */
template <typename DataT, int Dimension, typename PropT,
          typename GetAccFunctorT>
void check_has_property_member_func(GetAccFunctorT construct_acc,
                                    const sycl::range<Dimension> r) {
  DataT some_data(expected_val);
  {
    sycl::buffer<DataT, Dimension> data_buf(&some_data, r);
    auto accessor = construct_acc(data_buf);
    CHECK(accessor.template has_property<PropT>());
  }
}

/**
 * @brief Function invokes \c get_property() member function with \c PropT
 * property and verifies that returned object has same type as \c PropT
 *
 * @tparam GetAccFunctorT Type of functor that constructs testing accessor
 */
template <typename DataT, int Dimension, typename PropT,
          typename GetAccFunctorT>
void check_get_property_member_func(GetAccFunctorT construct_acc,
                                    const sycl::range<Dimension> r) {
  DataT some_data(expected_val);
  {
    sycl::buffer<DataT, Dimension> data_buf(&some_data, r);
    auto accessor = construct_acc(data_buf);
    auto acc_prop = accessor.template get_property<PropT>();

    CHECK(std::is_same_v<PropT, decltype(acc_prop)>);
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
decltype(auto) get_subscript_overload(AccT& accessor, size_t index) {
  if constexpr (dims == 1) return accessor[index];
  if constexpr (dims == 2) return accessor[index][index];
  if constexpr (dims == 3) return accessor[index][index][index];
}

}  // namespace accessor_tests_common

#endif  // SYCL_CTS_ACCESSOR_COMMON_H

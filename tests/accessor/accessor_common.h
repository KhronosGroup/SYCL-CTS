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
  std::string name = "Test ";
  name += section_description;
  name += " with parameters: <";
  name += type_name;
  name += "><";
  name += access_mode_name;
  name += "><";
  name += target_name;
  name += "><";
  name += std::to_string(Dimension) + ">";
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
  {
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
  }

  for (size_t i = 0; i < conditions_checks_size; i++) {
    CHECK(conditions_check[i]);
  }
}

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
 * @tparam GetAccFunctorT Type of functor for accessor creation
 */
template <accessor_type AccType, typename DataT, sycl::access_mode AccessMode,
          sycl::target Target, typename GetAccFunctorT>
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
          if constexpr (Target == sycl::target::host_task) {
            cgh.host_task([=] {
              read_write_zero_dim_acc<DataT, AccessMode>(acc, res_acc);
            });
          } else if constexpr (Target == sycl::target::device) {
            cgh.parallel_for_work_group(r, [=](sycl::group<1>) {
              read_write_zero_dim_acc<DataT, AccessMode>(acc, res_acc);
            });
          }
        })
        .wait_and_throw();
  }

  if constexpr (AccessMode != sycl::access_mode::write) {
    CHECK(compare_res);
  }
  if constexpr (AccessMode != sycl::access_mode::read) {
    CHECK(value_operations::are_equal(some_data, changed_val));
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
 * @tparam GetAccFunctorT Type of functor for accessor creation
 * @param r Range for accessors buffer
 */
template <accessor_type AccType, typename DataT, int Dimension,
          sycl::access_mode AccessMode, sycl::target Target,
          typename GetAccFunctorT>
void check_common_constructor(GetAccFunctorT get_accessor_functor,
                              const sycl::range<Dimension> r) {
  auto queue = util::get_cts_object::queue();
  bool compare_res = false;
  DataT some_data(expected_val);
  {
    sycl::buffer res_buf(&compare_res, sycl::range(1));
    sycl::buffer<DataT, Dimension> data_buf(&some_data, r);

    queue
        .submit([&](sycl::handler& cgh) {
          sycl::accessor res_acc(res_buf);
          auto acc = get_accessor_functor(data_buf, cgh);

          if (acc.is_placeholder()) {
            cgh.require(acc);
          }

          if constexpr (Target == sycl::target::host_task) {
            cgh.host_task([=] {
              read_write_acc<DataT, Dimension, AccessMode>(acc, res_acc);
            });
          } else if constexpr (Target == sycl::target::device) {
            cgh.parallel_for_work_group(sycl::range(1), [=](sycl::group<1>) {
              read_write_acc<DataT, Dimension, AccessMode>(acc, res_acc);
            });
          }
        })
        .wait_and_throw();
  }

  if constexpr (AccessMode != sycl::access_mode::write) {
    CHECK(compare_res);
  }
  if constexpr (AccessMode != sycl::access_mode::read) {
    CHECK(value_operations::are_equal(some_data, changed_val));
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
 * @tparam GetAccFunctorT Type of functor for accessor creation
 * @param r Range for accessors buffer
 */
template <accessor_type AccType, typename DataT, int Dimension,
          sycl::access_mode AccessMode, sycl::target Target,
          typename GetAccFunctorT>
void check_placeholder_accessor_exception(GetAccFunctorT get_accessor_functor,
                                          const sycl::range<Dimension> r) {
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
}  // namespace accessor_tests_common

#endif  // SYCL_CTS_ACCESSOR_COMMON_H

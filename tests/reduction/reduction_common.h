/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides common code for reduction tests
//
*******************************************************************************/

#ifndef __SYCL_CTS_TEST_REDUCTION_COMMON_H
#define __SYCL_CTS_TEST_REDUCTION_COMMON_H

#include "../common/common.h"
#include "../common/type_coverage.h"
// to use size_t
#include <cstddef>

namespace reduction_common {

using with_property = std::true_type;
using without_property = std::false_type;

static constexpr size_t number_iterations{10};

static constexpr int identity_value = 5;

constexpr int init_value_without_property_case{99};

static sycl::range<1> range{number_iterations};
static sycl::nd_range<1> nd_range{range, range};

static const auto scalar_types =
    named_type_pack<char, signed char, unsigned char, short int,
                    unsigned short int, int, unsigned int, long int,
                    unsigned long int, float, long long int,
                    unsigned long long int>(
        {"char", "signed char", "unsigned char", "short int",
         "unsigned short int", "int", "unsigned int", "long int",
         "unsigned long int", "float", "long long int",
         "unsigned long long int"});

/** @brief Returns expected value for testing
 *  @tparam VariableT The type of the variable with which the test runs
 *  @tparam FunctorT The type of the functor with which the test runs
 *  @tparam BufferT The type of the buffer with which the test runs
 *  @param functor The functor (plus, multiplies, etc.) with which the test runs
 *  @param buffer The buffer that will used in parallel_for() function
 *  @param value_for_initialization The value for initialization output variable
 *  @retval Expected value after test execution
 */
template <typename VariableT, typename FunctorT, typename BufferT>
VariableT get_expected_value(FunctorT functor, BufferT& buffer,
                             VariableT value_for_initialization) {
  VariableT expected_value{value_for_initialization};
  auto buf_accessor =
      buffer.template get_access<sycl::access_mode::read>(range);
  for (size_t i = 0; i < number_iterations; i++) {
    expected_value = functor(buf_accessor[i], expected_value);
  }
  return expected_value;
}

/** @brief This function contained common cases from two get_init_value*
 *         functions using property cases and functor type
 *  @tparam VariableT Variable type from type coverage
 *  @tparam FunctorT The type of the functor with which the test runs
 *  @tparam UsePropertyFlagT UseCombineFlagT std::integral_constant type that
 *          let switch between using and don't using
 *          sycl::property::reduction::initialize_to_identity
 *  @retval Value for initialization
 */
template <typename VariableT, typename FunctorT, typename UsePropertyFlagT>
void get_init_value_common_logic(VariableT& init_value) {
  if constexpr (UsePropertyFlagT::value &&
                (std::is_same_v<FunctorT, sycl::bit_and<VariableT>> ||
                 std::is_same_v<FunctorT, sycl::bit_or<VariableT>> ||
                 std::is_same_v<FunctorT, sycl::bit_xor<VariableT>>)) {
    init_value = 0x87654321;
  } else if constexpr (std::is_same_v<VariableT, bool> &&
                       UsePropertyFlagT::value) {
    init_value = !sycl::known_identity<FunctorT, VariableT>::value;
  } else if constexpr (std::is_same_v<VariableT, bool> &&
                       !UsePropertyFlagT::value) {
    init_value = sycl::known_identity<FunctorT, VariableT>::value;
  }
}

/** @brief Initialize value for reductions. If UsePropertyFlagT is true and
 *         sycl::has_known_identity for current functor and variable type is
 *         true then we use value that will not be equal to default reduction
 *         value and this value should be ignored if we don't use property or
 *         sycl::has_known_identity is false then we use common value with
 *         value that will be use in expected value initializations
 *  @tparam VariableT Variable type from type coverage
 *  @tparam FunctorT The type of the functor with which the test runs
 *  @tparam UsePropertyFlagT UseCombineFlagT std::integral_constant type that
 *          let switch between using and don't using
 *          sycl::property::reduction::initialize_to_identity
 *  @retval Value for reduction initialization
 */
template <typename VariableT, typename FunctorT,
          typename UsePropertyFlagT = std::false_type>
VariableT get_init_value_for_reduction() {
  VariableT init_value{};
  if constexpr (sycl::has_known_identity<FunctorT, VariableT>::value &&
                UsePropertyFlagT::value) {
    init_value = 50;
  } else {
    init_value = init_value_without_property_case;
  }
  get_init_value_common_logic<VariableT, FunctorT, UsePropertyFlagT>(
      init_value);
  return init_value;
}

/** @brief Initialize value for calculating expected value. If UsePropertyFlagT
 *         is true and sycl::has_known_identity for current functor and variable
 *         type is true then we use value that will not be equal to default
 *         reduction value and this value should be ignored if we don't use
 *         property or sycl::has_known_identity is false then we use common
 *         value with value that will be used in value for reductions
 *         initializations
 *  @tparam VariableT Variable type from type coverage
 *  @tparam FunctorT The type of the functor with which the test runs
 *  @tparam UsePropertyFlagT UseCombineFlagT std::integral_constant type that
 *          let switch between using and don't using
 *          sycl::property::reduction::initialize_to_identity
 *  @retval Value for expected value initialization
 */
template <typename VariableT, typename FunctorT,
          typename UsePropertyFlagT = std::false_type>
VariableT get_init_value_for_expected_value() {
  VariableT init_value{};
  if constexpr (sycl::has_known_identity<FunctorT, VariableT>::value &&
                UsePropertyFlagT::value) {
    // case when using reduction with initialize_to_identity
    init_value = sycl::known_identity<FunctorT, VariableT>::value;
  } else {
    init_value = init_value_without_property_case;
  }
  get_init_value_common_logic<VariableT, FunctorT, UsePropertyFlagT>(
      init_value);
  return init_value;
}

/** @brief Filling buffer for using it in parallel_for and computing
 * expected value
 *  @tparam VariableT Variable type from type coverage
 *  @tparam BufferT type sycl::buffer object
 *  @param buffer sycl::buffer object
 */
template <typename VariableT, typename BufferT>
void fill_buffer(BufferT& buffer) {
  sycl::host_accessor buf_accessor(buffer);

  // TODO: replace the loop with std::fill and std::iota
  bool value_for_filling_bool_buf{true};
  if (std::is_same_v<VariableT, bool>::value) {
    for (size_t i = 0; i < buffer.size(); i++) {
      buf_accessor[i] = value_for_filling_bool_buf;
    }
  } else {
    for (size_t i = 0; i < buffer.size(); i++) {
      buf_accessor[i] = i + 1;
    }
  }
}

/** @brief Construct new sycl::buffer object and fill it with default values
 *  @tparam VariableT Variable type from type coverage
 *  @retval Constructed and filled buffer
 */
template <typename VariableT>
auto get_buffer() {
  sycl::buffer<VariableT> initial_buf{range};
  fill_buffer<VariableT>(initial_buf);
  return initial_buf;
}

/** @brief Check that current device has sycl::aspect::usm_shared_allocations
 *         aspect
 *  @param queue sycl::queue class object
 *  @param log sycl_cts::util::logger class object
 *  @retval Bool value that corresponds to the presence of this aspect in the
 *          current device
 */
static bool check_usm_shared_aspect(sycl::queue& queue,
                                    sycl_cts::util::logger& log) {
  bool has_aspect =
      queue.get_device().has(sycl::aspect::usm_shared_allocations);
  if (!has_aspect) {
    log.note(
        "Device does not support accessing to unified shared memory "
        "allocation");
  }
  return has_aspect;
}

/** @brief A user-defined functor
 *  @tparam VariableT The type of the variable with which the test runs
 */
template <typename VariableT>
struct op_without_identity {
  VariableT operator()(const VariableT& x, const VariableT& y) const {
    return x + y;
  }
};

/** @brief A user-defined class with several scalar member variables, default
 *         constructor and some overloaded operators.
 */
struct custom_type {
  int m_int_field{};
  char m_char_field{};

  custom_type() = default;

  bool operator==(const custom_type& c_t) const {
    return m_int_field == c_t.m_int_field && m_char_field == c_t.m_char_field;
  }

  operator int() const { return m_int_field; }

  void operator=(int value) {
    m_int_field = value;
    m_char_field = value;
  }
};

static custom_type operator+(const custom_type& c_t_l,
                             const custom_type& c_t_r) {
  custom_type temp;
  temp.m_int_field = c_t_l.m_int_field + c_t_r.m_int_field;
  temp.m_char_field = c_t_l.m_char_field + c_t_r.m_char_field;
  return temp;
}

}  // namespace reduction_common

#endif  // __SYCL_CTS_TEST_REDUCTION_COMMON_H

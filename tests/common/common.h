/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2020-2022 The Khronos Group Inc.
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

#ifndef __SYCLCTS_TESTS_COMMON_COMMON_H
#define __SYCLCTS_TESTS_COMMON_COMMON_H

#include <sycl/sycl.hpp>

#include <catch2/catch_test_macros.hpp>

#include "../../util/conversion.h"
#include "../../util/math_vector.h"
#include "../../util/proxy.h"
#include "../../util/sycl_enums.h"
#include "../../util/test_base.h"

#include "cts_async_handler.h"
#include "cts_selector.h"
#include "get_cts_object.h"
#include "macros.h"
#include "string_makers.h"
#include "value_operations.h"

#include <cinttypes>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>

namespace {

/**
 * @brief Helper function to print an error message and fail a test
 */
inline void fail_test(sycl_cts::util::logger& log,
                      std::string errorMsg) {
  FAIL(log, errorMsg);
}

/**
 * @brief Helper function to check the return value of a function.
 *
 * @deprecated Prefer using CHECK/REQUIRE macros instead.
 */
template <typename T>
void check_return_value(sycl_cts::util::logger& log, const T& a, const T& b,
                        std::string functionName) {
  if (a != b) {
    FAIL(log, functionName + " returns an incorrect value");
  }
};

/**
 * @brief Helper function to check the return type of a function.
 */
template <typename ExpectedT, typename ReturnT>
void check_return_type(ReturnT returnVal, std::string functionName) {
  INFO(functionName + " has incorrect return type -> " +
       typeid(ReturnT).name());
  CHECK(std::is_same<ReturnT, ExpectedT>::value);
}

/**
 * @deprecated Use overload without logger.
 */
template <typename ExpectedT, typename ReturnT>
void check_return_type(sycl_cts::util::logger& log, ReturnT returnVal,
                       std::string functionName) {
  check_return_type<ExpectedT>(returnVal, functionName);
}

/**
 * @brief Helper function to check the return type of a function.
 */
template <typename ExpectedT, typename ReturnT>
bool check_return_type_bool(ReturnT returnVal) {
  return std::is_same<ExpectedT, ReturnT>::value;
}

/**
 * @brief Helper function to check two types are equal.
 */
template <typename ExpectedT, typename ActualT>
void check_equal_type(ActualT actualVal, std::string logMsg) {
  if (typeid(ExpectedT) != typeid(ActualT)) {
    FAIL(logMsg << "\nGot type -> " << typeid(ActualT).name()
                << "\nExpected type -> " << typeid(ExpectedT).name());
  }
}

/**
 * @deprecated Use overload without logger.
 */
template <typename ExpectedT, typename ActualT>
void check_equal_type(sycl_cts::util::logger& log, ActualT actualVal,
                      std::string logMsg) {
  check_equal_type<ExpectedT>(actualVal, logMsg);
}

/**
 * @brief Helper function to check two types are equal.
 */
template <typename ExpectedT, typename ActualT>
bool check_equal_type_bool(ActualT actualVal) {
  return std::is_same<ExpectedT, ActualT>::value;
}

/**
 * @brief Helper function to check for the existence of an enum class value.
 */
template <typename enumT>
void check_enum_class_value(enumT value) {
  enumT tmp = value;
}

/**
 * @brief Helper function to check an enum is of the correct underlying type.
 */
template <typename enumT, typename underlyingT>
void check_enum_underlying_type(sycl_cts::util::logger& log) {
  if (typeid(typename std::underlying_type<enumT>::type) !=
      typeid(underlyingT)) {
    FAIL(log, std::string(
                  typeid(typename std::underlying_type<enumT>::type).name()) +
                  " enum underlying type is not " +
                  std::string(typeid(underlyingT).name()));
  }
}

/**
 * @brief Helper function to check an info parameter.
 */
template <typename InfoDesc, typename ReturnT, typename ObjectT>
void check_get_info_param(const ObjectT& object) {
  // Check return_type specified in the descriptor
  INFO("Information descriptor has incorrect return_type");
  CHECK(std::is_same_v<typename InfoDesc::return_type, ReturnT>);

  // Check get_info return type
  auto returnValue = object.template get_info<InfoDesc>();
  check_return_type<ReturnT>(returnValue, "object::get_info()");
}

/**
 * @deprecated Use overload without logger.
 */
template <typename InfoDesc, typename ReturnT, typename ObjectT>
void check_get_info_param(sycl_cts::util::logger& log, const ObjectT& object) {
  check_get_info_param<InfoDesc, ReturnT>(object);
}

/**
 * @brief Helper function to check the equality of two SYCL objects.
 */
template <typename T>
void check_equality(T& a, T& b) {
  /** check get_backend
   */
  if (a.get_backend() != b.get_backend()) {
    FAIL("two objects are not equal (get_backend)");
  }

#ifdef SYCL_BACKEND_OPENCL
  /** check get_native
   */
  auto queue = sycl_cts::util::get_cts_object::queue();
  if (queue.get_backend() == sycl::backend::opencl) {
#if !SYCL_CTS_COMPILING_WITH_COMPUTECPP
    if (sycl::get_native<sycl::backend::opencl>(a) !=
        sycl::get_native<sycl::backend::opencl>(b)) {
#else
    if (a.get() != b.get()) {
#endif
      FAIL("two objects are not equal");
    }
  }
#endif  // SYCL_BACKEND_OPENCL
};

/**
 * @deprecated Use overload without logger.
 */
template <typename T>
void check_equality(sycl_cts::util::logger& log, T& a, T& b) {
  check_equality(a, b);
};

/**
 * @brief Helper function to test two arrays have equal elements. Deprecated.
 * Use \c value_operations::are_equal instead
 */
template <typename ArrT, int size>
void check_array_equality(ArrT* arr1, ArrT* arr2) {
  for (int i = 0; i < size; i++) {
    if (arr1[i] != arr2[i]) {
      FAIL("arrays are not equal");
    }
  }
}

/**
 * @deprecated Use overload without logger. Deprecated.
 * Use \c value_operations::are_equal instead
 */
template <typename ArrT, int size>
void check_array_equality(sycl_cts::util::logger& log, ArrT* arr1, ArrT* arr2) {
  check_array_equality<ArrT, size>(arr1, arr2);
}

/**
 * @brief Helper function to see if a type is of the wrong size
 */
template <typename T>
bool check_type_min_size(size_t minSize) {
  return !(sizeof(T) < minSize);
}

/**
 * @brief Helper function to see if a type is of the wrong sign
 */
template <typename T>
bool check_type_sign(bool expected_sign) {
  return (std::is_signed<T>::value == expected_sign);
}

/**
 * @brief Helper function to see if sycl::half is of the wrong sign
 */
template <>
inline bool check_type_sign<sycl::half>(bool expected_sign) {
  bool is_signed = sycl::half(1) > sycl::half(-1);
  return is_signed == expected_sign;
}

/**
 * @brief Helper function to log a failure if a type is of the wrong size or
 * sign
 */
template <typename T>
void check_type_min_size_sign_log(size_t minSize, bool expected_sign,
                                  std::string typeName) {
  if (!check_type_min_size<T>(minSize)) {
    FAIL(std::string(
             "The following host type does not have the correct size: ") +
         typeName);
  }
  if (!check_type_sign<T>(expected_sign)) {
    FAIL(std::string(
             "The following host type does not have the correct sign: ") +
         typeName);
  }
}

/**
 * @deprecated Use overload without logger.
 */
template <typename T>
void check_type_min_size_sign_log(sycl_cts::util::logger& log, size_t minSize,
                                  bool expected_sign, std::string typeName) {
  check_type_min_size_sign_log<T>(minSize, expected_sign, typeName);
}

/**
 * @brief Verify two values are equal. Deprecated.
 * Use \c value_operations::are_equal instead
 */
template <typename T>
bool check_equal_values(const T& lhs, const T& rhs) {
  return value_operations::are_equal(lhs, rhs);
}

/**
 * @brief Instantiation for vectors with the same API as for scalar values. 
 * Deprecated. Use \c value_operations::are_equal instead
 */
template <typename T, int numElements>
bool check_equal_values(const sycl::vec<T, numElements>& lhs,
                        const sycl::vec<T, numElements>& rhs) {
  bool result = true;
  auto perElement = lhs == rhs;
  for (int i = 0; i < numElements; ++i) {
    result &= perElement[i] != 0;
  }
  return result;
}

// ComputeCpp and hipSYCL do not yet support sycl::marray
#if !SYCL_CTS_COMPILING_WITH_COMPUTECPP && !SYCL_CTS_COMPILING_WITH_HIPSYCL
/**
 * @brief Instantiation for marray with the same API as for scalar values
 * Deprecated. Use \c value_operations::are_equal instead
 */
template <typename T, std::size_t numElements>
bool check_equal_values(const sycl::marray<T, numElements>& lhs,
                        const sycl::marray<T, numElements>& rhs) {
  auto perElement = lhs == rhs;
  return std::all_of(perElement.begin(), perElement.end(), [](bool el){
    return el;
  });
}
#endif

/** Enables concept checking ahead of the Concepts TS
 *  Idea for macro taken from Eric Niebler's range-v3
 */
#define REQUIRES_IMPL(B) typename std::enable_if<(B), int>::type = 1
#define REQUIRES(...) REQUIRES_IMPL((__VA_ARGS__))

/**
 * @brief Transforms the input type into a dependant type and performs
 *        an enable_if based on the condition.
 *
 *        Useful for disabling a member function based on a template parameter
 *        of the class.
 * @param typeName Non-dependant type to be made dependant
 * @param condition The condition specifying when the template
 *        should be enabled. typeName can occur within this expression.
 */
#define ENABLE_IF_DEPENDANT(typeName, condition)                               \
  typename overloadDependantT = typeName,                                      \
           typename = typename std::enable_if <                                \
                          std::is_same<typeName, overloadDependantT>::value && \
                      (condition) > ::type

template <bool condition, typename F1, typename F2,
          bool same_return_type =
              std::is_same<typename std::result_of<F1&()>::type,
                           typename std::result_of<F2&()>::type>::value>
struct if_constexpr_impl;

template <bool condition, typename F1, typename F2>
struct if_constexpr_impl<condition, F1, F2, true> {
  static constexpr auto result(const F1& f1, const F2& f2) -> decltype(f1()) {
    return condition ? f1() : f2();
  }
};

template <typename F1, typename F2>
struct if_constexpr_impl<true, F1, F2, false> {
  static constexpr auto result(const F1& f1, const F2&) -> decltype(f1()) {
    return f1();
  }
};

template <typename F1, typename F2>
struct if_constexpr_impl<false, F1, F2, false> {
  static constexpr auto result(const F1&, const F2& f2) -> decltype(f2()) {
    return f2();
  }
};

/**
 * @brief Library implementation for C++17's compile-time if-statement so that
 * it works in C++11. Generates a call to the invocable object `f1` if
 * `condition == true` at compile-time, otherwise a call to `f2` is generated.
 */
template <bool condition, typename F1, typename F2,
          typename R = typename std::conditional<
              condition, typename std::result_of<F1&()>::type,
              typename std::result_of<F2&()>::type>::type>
inline R if_constexpr(const F1& f1, const F2& f2) {
  return if_constexpr_impl<condition, F1, F2>::result(f1, f2);
}

/**
 * @brief Library implementation for C++17's compile-time if-statement so that
 * it works in C++11. Generates a call to the invocable object `f` if
 * `condition == true` at compile-time, otherwise no code is generated.
 */
template <bool condition, typename F>
inline void if_constexpr(const F& f) {
  if (condition) {
    f();
  }
}

/**
 * @brief Tag to denote mapping of integer coordinates to real scale
 *
 * For example, if we make a one pixel wide image this pixel can have
 * any coordinate in range [0.0 .. 1.0)
 */
namespace pixel_tag {
  struct generic {};
  /** @brief The low boundary of the pixel, equal to the integer one
   *         if representable
   */
  struct lower : generic {};
  /** @brief The upper boundary of the pixel, equal to the left limit
   *         lim(x-) where x is the low boundary for the next pixel
   */
  struct upper: generic {};
};

// ComputeCpp does not yet support operator[] on sycl::vec
// hipSYCL does not yet support images
#if !SYCL_CTS_COMPILING_WITH_COMPUTECPP && !SYCL_CTS_COMPILING_WITH_HIPSYCL

/**
 * @brief Helps with retrieving the right access type for reading/writing
 *        an image
 * @tparam dims Number of image dimensions
 */
template <int dims>
struct image_access;

/**
 * @brief Specialization for one dimension
 */
template <>
struct image_access<1> {
  using int_type = sycl::cl_int;
  using float_type = sycl::cl_float;
  static int_type get_int(const sycl::id<1>& i) {
    return int_type(i.get(0));
  }
  static int_type get_int(const sycl::item<1>& i) {
    return get_int(i.get_id());
  }
  static float_type get_float(const sycl::id<1>& i) {
    return float_type(static_cast<float>(i.get(0)));
  }
  static float_type get_float(const sycl::item<1>& i) {
    return get_float(i.get_id());
  }
  static float_type get_normalized(const pixel_tag::lower,
                                   const sycl::id<1>& i,
                                   const sycl::range<1>& r) {
    return get_float(i) / static_cast<int>(r.get(0));
  }
  static float_type get_normalized(const pixel_tag::upper,
                                   const sycl::id<1>& i,
                                   const sycl::range<1>& r) {
    const auto negative_inf =
        -1.0f * std::numeric_limits<float_type>::infinity();
    const auto next = get_normalized(pixel_tag::lower{}, 1 + i, r);

    return sycl::nextafter(next, negative_inf);
  }
};

/**
 * @brief Specialization for two dimensions
 */
template <>
struct image_access<2> {
  using int_type = sycl::cl_int2;
  using float_type = sycl::cl_float2;
  static int_type get_int(const sycl::id<2>& i) {
    return int_type(i.get(0), i.get(1));
  }
  static int_type get_int(const sycl::item<2>& i) {
    return get_int(i.get_id());
  }
  static float_type get_float(const sycl::id<2>& i) {
    return float_type(static_cast<float>(i.get(0)),
                      static_cast<float>(i.get(1)));
  }
  static float_type get_float(const sycl::item<2>& i) {
    return get_float(i.get_id());
  }
  static float_type get_normalized(const pixel_tag::lower,
                                   const sycl::id<2>& i,
                                   const sycl::range<2>& r) {
    return float_type(
        static_cast<float>(i.get(0)) / static_cast<int>(r.get(0)),
        static_cast<float>(i.get(1)) / static_cast<int>(r.get(1)));
  }
  static float_type get_normalized(const pixel_tag::upper,
                                   const sycl::id<2>& i,
                                   const sycl::range<2>& r) {
    const auto negative_inf = -1.0f * std::numeric_limits<float>::infinity();
    const auto next = get_normalized(pixel_tag::lower{}, 1 + i, r);

    return float_type(sycl::nextafter(next[0], negative_inf),
                      sycl::nextafter(next[1], negative_inf));
  }
};

/**
 * @brief Specialization for three dimensions
 */
template <>
struct image_access<3> {
  using int_type = sycl::cl_int4;
  using float_type = sycl::cl_float4;
  static int_type get_int(const sycl::id<3>& i) {
    return int_type(i.get(0), i.get(1), i.get(2), 0);
  }
  static int_type get_int(const sycl::item<3>& i) {
    return get_int(i.get_id());
  }
  static float_type get_float(const sycl::id<3>& i) {
    return float_type(static_cast<float>(i.get(0)),
                      static_cast<float>(i.get(1)),
                      static_cast<float>(i.get(2)), .0f);
  }
  static float_type get_float(const sycl::item<3>& i) {
    return get_float(i.get_id());
  }
  static float_type get_normalized(const pixel_tag::lower,
                                   const sycl::id<3>& i,
                                   const sycl::range<3>& r) {
    return float_type(
        static_cast<float>(i.get(0)) / static_cast<int>(r.get(0)),
        static_cast<float>(i.get(1)) / static_cast<int>(r.get(1)),
        static_cast<float>(i.get(2)) / static_cast<int>(r.get(2)), .0f);
  }
  static float_type get_normalized(const pixel_tag::upper,
                                   const sycl::id<3>& i,
                                   const sycl::range<3>& r) {
    const auto negative_inf = -1.0f * std::numeric_limits<float>::infinity();
    const auto next = get_normalized(pixel_tag::lower{}, 1 + i, r);

    return float_type(sycl::nextafter(next[0], negative_inf),
                      sycl::nextafter(next[1], negative_inf),
                      sycl::nextafter(next[2], negative_inf), .0f);
  }
};

#endif

/**
 * @brief Dummy template function to check type existence without generating warnings.
 */
template <typename T>
void constexpr check_type_existence() {
};


/**
 * @brief Helper function to check if all devices support online compiler.
 */
inline bool is_compiler_available(
    const std::vector<sycl::device>& deviceList) {
  bool compiler_available = true;
  for (const auto& device : deviceList) {
    if (!device.get_info<sycl::info::device::is_compiler_available>()) {
      compiler_available = false;
      break;
    }
  }
  return compiler_available;
}

/**
 * @brief Helper function to check if all devices support online linker.
 */
inline bool is_linker_available(
    const std::vector<sycl::device>& deviceList) {
  bool linker_available = true;
  for (const auto& device : deviceList) {
    if (!device.get_info<sycl::info::device::is_linker_available>()) {
      linker_available = false;
      break;
    }
  }
  return linker_available;
}

/**
 * @brief Helper function to check work-group size device limit
 * @param log Logger to use
 * @param queue Queue to verify against
 * @param wgSize Work-group size to verify for support
 */
inline bool device_supports_wg_size(sycl_cts::util::logger& log,
                                    sycl::queue &queue,
                                    size_t wgSize)
{
  auto device = queue.get_device();
  const auto maxDeviceWorkGroupSize =
      device.template get_info<sycl::info::device::max_work_group_size>();

  const bool supports = maxDeviceWorkGroupSize >= wgSize;
  if (!supports)
    log.note("Device does not support work group size %" PRIu64,
             static_cast<std::uint64_t>(wgSize));
  return supports;
}

/**
 * @brief Helper function to check work-group size kernel limit
 * @tparam kernelT Kernel to run onto
 * @param log Logger to use
 * @param queue Queue to verify against
 * @param wgSize Work-group size to verify for support
 */
template <class kernelT>
inline bool kernel_supports_wg_size(sycl_cts::util::logger& log,
                                    sycl::queue &queue,
                                    size_t wgSize)
{
  // Verify only for device in use
  auto device = queue.get_device();
  const auto& context = queue.get_context();
  const std::vector<sycl::device> devicesToCheck{device};

  /* To query info::kernel_work_group::work_group_size property, we need to
   * obtain test kernel handler, which requires online compilation
   * */
  if (!is_compiler_available(devicesToCheck) ||
      !is_linker_available(devicesToCheck)) {
    log.note("Device does not support online compilation");
    return false;
  }

// ComputeCpp and hipSYCL do not yet support sycl::get_kernel_bundle
#if !SYCL_CTS_COMPILING_WITH_COMPUTECPP && !SYCL_CTS_COMPILING_WITH_HIPSYCL
  auto kb =
      sycl::get_kernel_bundle<kernelT, sycl::bundle_state::executable>(context);
  auto kernel = kb.get_kernel(sycl::get_kernel_id<kernelT>());
#else
  sycl::program program(context, devicesToCheck);
  program.build_with_kernel_type<kernelT>("");
  auto kernel = program.get_kernel<kernelT>();
#endif
  auto maxKernelWorkGroupSize =
      device.template get_info<sycl::info::device::max_work_group_size>();

  const bool supports = maxKernelWorkGroupSize >= wgSize;
  if (!supports) {
    // We cannot use %zu in C++11; see P0330R8 proposal
    log.note("Kernel does not support work group size %" PRIu64,
             static_cast<std::uint64_t>(wgSize));
  }
  return supports;
}

}  // namespace

/** \brief tests the result of using operator op with operands lhs and rhs,
 * while storing the results in res.
 */
#define INDEX_KERNEL_TEST(op, lhs, rhs, res)                               \
  {                                                                        \
    res = (lhs op rhs);                                                    \
    for (int k = 0; k < dims; k++) {                                       \
      if ((res.get(k) != static_cast<size_t>(lhs.get(k) op rhs.get(k))) || \
          (res[k] != static_cast<size_t>(lhs[k] op rhs[k]))) {             \
        error_ptr[m_iteration] = __LINE__;                                 \
        m_iteration++;                                                     \
      }                                                                    \
    }                                                                      \
  }

/** \brief tests the result of equality/inequality operator op between INDEX
 * operands lhs and rhs
 */
#define INDEX_EQ_KERNEL_TEST(op, lhs, rhs)          \
  {                                                 \
    if ((lhs op lhs) != (rhs op rhs)) {             \
      error_ptr[m_iteration] = __LINE__;            \
      m_iteration++;                                \
    }                                               \
    bool result = lhs op rhs;                       \
    for (int k = 0; k < dims; k++) {                \
      if ((result != (lhs.get(k) op rhs.get(k))) || \
          (result != (lhs[k] op rhs[k]))) {         \
        error_ptr[m_iteration] = __LINE__;          \
        m_iteration++;                              \
      }                                             \
    }                                               \
  }

/** \brief tests the result of operator op between scalar operand lhs and INDEX
 * operand rhs
 */
#define INDEX_SIZE_T_KERNEL_TEST(op, INDEX, integer, result)                 \
  {                                                                          \
    result = INDEX op integer;                                               \
    for (int k = 0; k < dims; k++) {                                         \
      if (result.get(k) != (static_cast<size_t>(INDEX.get(k) op integer)) || \
          (result[k] != static_cast<size_t>(INDEX[k] op integer))) {         \
        error_ptr[m_iteration] = __LINE__;                                   \
        m_iteration++;                                                       \
      }                                                                      \
    }                                                                        \
  }

/** \brief tests the result of operator op between scalar operand lhs and INDEX
 * operand rhs
 */
#define SIZE_T_INDEX_KERNEL_TEST(op, integer, INDEX, result)                 \
  {                                                                          \
    result = integer op INDEX;                                               \
    for (int k = 0; k < dims; k++) {                                         \
      if (result.get(k) != (static_cast<size_t>(integer op INDEX.get(k))) || \
          (result[k] != static_cast<size_t>(integer op INDEX[k]))) {         \
        error_ptr[m_iteration] = __LINE__;                                   \
        m_iteration++;                                                       \
      }                                                                      \
    }                                                                        \
  }

/** \brief tests the result of operator \p op between \p integer operand and an
 * \p INDEX operand in any possible configuration
 */
#define DUAL_SIZE_INDEX_KERNEL_TEST(op, INDEX, integer, result) \
  INDEX_SIZE_T_KERNEL_TEST(op, INDEX, integer, result);         \
  SIZE_T_INDEX_KERNEL_TEST(op, integer, INDEX, result)

/** \brief tests the result of assignment operator \p op between assigning \p a
 * to \p c then use the assignment operator \p assignment_op with lhs operand \p
 * c and rhs operand \p b. Then tests the result using operator \p op with
 * operands \p a and \p b.
 */
#define INDEX_ASSIGNMENT_TESTS(assignment_op, op, a, b, c)                    \
  {                                                                           \
    c = a;                                                                    \
    c assignment_op b;                                                        \
    for (int k = 0; k < dims; k++) {                                          \
      if ((c.get(k) != (a.get(k) op b.get(k))) || (c[k] != (a[k] op b[k]))) { \
        error_ptr[m_iteration] = __LINE__;                                    \
        m_iteration++;                                                        \
      }                                                                       \
    }                                                                         \
  }

/** \brief tests the result of assignment operator \p op between assigning \p a
 * to \p c then use the assignment operator \p assignment_op with lhs operand \p
 * c and rhs operand \p integer. Then tests the result using operator \p op with
 * operands \p a and \p integer.
 */
#define INDEX_ASSIGNMENT_INTEGER_TESTS(assignment_op, op, a, integer, c) \
  {                                                                      \
    c = a;                                                               \
    c assignment_op integer;                                             \
    for (int k = 0; k < dims; k++) {                                     \
      if ((c.get(k) != (a.get(k) op integer)) ||                         \
          (c[k] != (a[k] op integer))) {                                 \
        error_ptr[m_iteration] = __LINE__;                               \
        m_iteration++;                                                   \
      }                                                                  \
    }                                                                    \
  }
#endif  // __SYCLCTS_TESTS_COMMON_COMMON_H

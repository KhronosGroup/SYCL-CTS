/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_COMMON_COMMON_H
#define __SYCLCTS_TESTS_COMMON_COMMON_H

// include our proxy to the real sycl header
#include "sycl.h"

// test framework specific device selector
#include "../common/cts_selector.h"
#include "../common/cts_async_handler.h"
#include "../common/get_cts_object.h"
#include "../../util/proxy.h"
#include "macros.h"

#include "../../util/opencl_helper.h"
#include "../../util/test_base.h"
#include "../../util/test_base_opencl.h"
#include "../../util/math_vector.h"

#include <string>
#include <sstream>
#include <type_traits>
#include <numeric>

namespace {

/**
 * @brief Helper function to print an error message and fail a test
 */
inline void fail_test(sycl_cts::util::logger& log,
                      cl::sycl::string_class errorMsg) {
  FAIL(log, errorMsg);
}

/**
 * @brief Helper function to check the return value of a function.
 */
template <typename T>
void check_return_value(sycl_cts::util::logger& log, const T& a, const T& b,
                        cl::sycl::string_class functionName) {
  if (a != b) {
    FAIL(log, functionName + " returns an incorrect value");
  }
};

/**
 * @brief Helper function to check the return type of a function.
 */
template <typename expectedT, typename returnT>
void check_return_type(sycl_cts::util::logger& log, returnT returnVal,
                       cl::sycl::string_class functionName) {
  if (!std::is_same<returnT, expectedT>::value) {
    FAIL(log, functionName + " has incorrect return type -> " +
                  cl::sycl::string_class(typeid(returnT).name()));
  }
}

/**
 * @brief Helper function to check the return type of a function.
 */
template <typename expectedT, typename returnT>
bool check_return_type_bool(returnT returnVal) {
  return std::is_same<expectedT, returnT>::value;
}

/**
 * @brief Helper function to check two types are equal.
 */
template <typename expectedT, typename actualT>
void check_equal_type(sycl_cts::util::logger& log, actualT actualVal,
                      cl::sycl::string_class logMsg) {
  if (typeid(expectedT) != typeid(actualT)) {
    FAIL(log, logMsg + "\nGot type -> " +
                  cl::sycl::string_class(typeid(actualT).name()) +
                  "\nExpected type -> " +
                  cl::sycl::string_class(typeid(expectedT).name()));
  }
}

/**
 * @brief Helper function to check two types are equal.
 */
template <typename expectedT, typename actualT>
bool check_equal_type_bool(actualT actualVal) {
  return std::is_same<expectedT, actualT>::value;
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
    FAIL(log, cl::sycl::string_class(
                  typeid(typename std::underlying_type<enumT>::type).name()) +
                  " enum underlying type is not " +
                  cl::sycl::string_class(typeid(underlyingT).name()));
  }
}

/**
 * @brief Helper function to check an info parameter.
 */
template <typename enumT, typename returnT, enumT kValue, typename objectT>
void check_get_info_param(sycl_cts::util::logger& log, const objectT& object) {
  /** check param_traits return type
   */
  using paramTraitsType =
      typename cl::sycl::info::param_traits<enumT, kValue>::return_type;
  if (typeid(paramTraitsType) != typeid(returnT)) {
    FAIL(log, "param_trait specialization has incorrect return type");
  }

  /** check get_info return type
   */
  auto returnValue = object.template get_info<kValue>();
  check_return_type<returnT>(log, returnValue, "object::get_info()");
}

/**
 * @brief Helper function to check a profiling info parameter.
 */
template <typename enumT, typename returnT, enumT kValue, typename objectT>
void check_get_profiling_info_param(sycl_cts::util::logger& log,
                                    const objectT& object) {
  /** check param_traits return type
   */
  using paramTraitsType =
      typename cl::sycl::info::param_traits<enumT, kValue>::return_type;
  if (!std::is_same<paramTraitsType, returnT>::value) {
    FAIL(log, "param_trait specialization has incorrect return type");
  }

  /** check get_profiling_info return type
   */
  auto returnValue = object.template get_profiling_info<kValue>();
  check_return_type<returnT>(log, returnValue, "object::get_profiling_info()");
}

/**
 * @brief Helper function to check the equality of two SYCL objects.
 */
template <typename T>
void check_equality(sycl_cts::util::logger& log, T& a, T& b, bool checkGet) {
  /** check is_host
   */
  if (a.is_host() != b.is_host()) {
    FAIL(log, "two objects are not equal (is_host)");
  }

  /** check get
   */
  if (checkGet) {
    if (a.get() != b.get()) {
      FAIL(log, "two objects are not equal (get)");
    }
  }
};

/**
 * @brief Helper function to test two arrays have equal elements
 */
template <typename arrT, int size>
void check_array_equality(sycl_cts::util::logger& log, arrT* arr1, arrT* arr2) {
  for (int i = 0; i < size; i++) {
    if (arr1[i] != arr2[i]) {
      FAIL(log, "arrays are not equal");
    }
  }
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
 * @brief Helper function to see if cl::sycl::half is of the wrong sign
 */
template <>
inline bool check_type_sign<cl::sycl::half>(bool expected_sign) {
  bool is_signed = cl::sycl::half(1) > cl::sycl::half(-1);
  return is_signed == expected_sign;
}

/**
 * @brief Helper function to log a failure if a type is of the wrong size or
 * sign
 */
template <typename T>
void check_type_min_size_sign_log(sycl_cts::util::logger& log, size_t minSize,
                                  bool expected_sign,
                                  cl::sycl::string_class typeName) {
  if (!check_type_min_size<T>(minSize)) {
    FAIL(log, cl::sycl::string_class(
                  "The following host type does not have the correct size: ") +
                  typeName);
  }
  if (!check_type_sign<T>(expected_sign)) {
    FAIL(log, cl::sycl::string_class(
                  "The following host type does not have the correct sign: ") +
                  typeName);
  }
}

/** helper function for retrieving an event from a submitted kernel
 */
template <typename kernelT>
cl::sycl::event get_queue_event(cl::sycl::queue& queue) {
  return queue.submit([&](cl::sycl::handler& handler) {
    handler.single_task<kernelT>([=]() {});
  });
}

#define TEST_TYPE_TRAIT(syclType, param, syclObject)                    \
  if (typeid(cl::sycl::info::param_traits<                              \
             cl::sycl::info::syclType,                                  \
             cl::sycl::info::syclType::param>::return_type) !=          \
      typeid(syclObject.get_info<cl::sycl::info::syclType::param>())) { \
    FAIL(log, #syclType                                                 \
         ".get_info<cl::sycl::info::" #syclType "::" #param             \
         ">() does not return "                                         \
         "cl::sycl::info::param_traits<cl::sycl::info::" #syclType      \
         ", cl::sycl::info::" #syclType "::" #param ">::return_type");  \
  }

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

template <typename T>
void check_equality_comparable_generic(sycl_cts::util::logger& log, const T& a,
                                       std::string test_name) {
  /** check for reflexivity
   */
  if (!(a == a)) {
    FAIL(log, (test_name +
               " is not equality-comparable (operator== reflexivity failed)")
                  .c_str());
  } else if (a != a) {
    FAIL(log, (test_name +
               " is not equality-comparable (operator!= reflexivity failed)")
                  .c_str());
  }

  /** check for symmetry
   */
  auto b = a;
  if (!(a == b)) {
    FAIL(log, (test_name +
               " is not equality-comparable (operator==, copy constructor)")
                  .c_str());
  } else if (!(b == a)) {
    FAIL(log, (test_name +
               " is not equality-comparable (operator== symmetry failed)")
                  .c_str());
  } else if (a != b) {
    FAIL(log, (test_name +
               " is not equality-comparable (operator!=, copy constructor)")
                  .c_str());
  } else if (b != a) {
    FAIL(log, (test_name +
               " is not equality-comparable (operator!= symmetry failed)")
                  .c_str());
  }

  /** check for transitivity
   */
  auto c = b;
  if (!(a == c)) {
    FAIL(log, (test_name +
               " is not equality-comparable (operator== transitivity failed)")
                  .c_str());
  } else if (a != c) {
    FAIL(log, (test_name +
               " is not equality-comparable (operator!= transitivity  failed)")
                  .c_str());
  }
}

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
  using int_type = cl::sycl::cl_int;
  using float_type = cl::sycl::cl_float;
  static int_type get_int(const cl::sycl::id<1>& i) {
    return int_type(i.get(0));
  }
  static int_type get_int(const cl::sycl::item<1>& i) {
    return get_int(i.get_id());
  }
  static float_type get_float(const cl::sycl::id<1>& i) {
    return float_type(static_cast<float>(i.get(0)));
  }
  static float_type get_float(const cl::sycl::item<1>& i) {
    return get_float(i.get_id());
  }
};

/**
 * @brief Specialization for two dimensions
 */
template <>
struct image_access<2> {
  using int_type = cl::sycl::cl_int2;
  using float_type = cl::sycl::cl_float2;
  static int_type get_int(const cl::sycl::id<2>& i) {
    return int_type(i.get(0), i.get(1));
  }
  static int_type get_int(const cl::sycl::item<2>& i) {
    return get_int(i.get_id());
  }
  static float_type get_float(const cl::sycl::id<2>& i) {
    return float_type(static_cast<float>(i.get(0)),
                      static_cast<float>(i.get(1)));
  }
  static float_type get_float(const cl::sycl::item<2>& i) {
    return get_float(i.get_id());
  }
};

/**
 * @brief Specialization for three dimensions
 */
template <>
struct image_access<3> {
  using int_type = cl::sycl::cl_int4;
  using float_type = cl::sycl::cl_float4;
  static int_type get_int(const cl::sycl::id<3>& i) {
    return int_type(i.get(0), i.get(1), i.get(2), 0);
  }
  static int_type get_int(const cl::sycl::item<3>& i) {
    return get_int(i.get_id());
  }
  static float_type get_float(const cl::sycl::id<3>& i) {
    return float_type(static_cast<float>(i.get(0)),
                      static_cast<float>(i.get(1)),
                      static_cast<float>(i.get(2)), .0f);
  }
  static float_type get_float(const cl::sycl::item<3>& i) {
    return get_float(i.get_id());
  }
};

/**
 * @brief Display the name of the type
 * @tparam T Type to inspect
 * @param Dummy parameter used only for type deduction
 * @return String representing the type
 */
template <typename T>
cl::sycl::string_class type_to_string(T) {
  using no_cv_t = typename std::remove_cv<T>::type;
  if (std::is_integral<no_cv_t>::value) {
    cl::sycl::string_class intStr;
    if (std::is_signed<no_cv_t>::value) {
      intStr = "int";
    } else {
      intStr = "uint";
    }
    const auto typeSizeBits = sizeof(T) * 8;
    return intStr + std::to_string(typeSizeBits) + "_t";
  } else if (std::is_same<no_cv_t, float>::value) {
    return "float";
  } else if (std::is_same<no_cv_t, double>::value) {
    return "double";
  } else if (std::is_same<no_cv_t, cl::sycl::half>::value) {
    return "half";
  }
  return "unknown";
}

/**
 * @brief Display the name of the vec type
 * @tparam dataT Underlying type of the vector
 * @tparam numElems Number of elements of the vector
 * @param Dummy parameter used only for type deduction
 * @return String representing the vec type
 */
template <typename dataT, int numElems>
cl::sycl::string_class type_to_string(cl::sycl::vec<dataT, numElems>) {
  std::stringstream stream;
  stream << "vec<" << type_to_string(dataT{}) << ", " << numElems << ">";
  return stream.str();
}

/**
 * @brief Dummy template to check type existence without generating warnings.
 */
template <typename T>
struct check_type_existence {
  check_type_existence() = default;
};

/** Helper function that calculates a range from a size so
 *  that each dimension equals size
 */
template <size_t dims>
cl::sycl::range<dims> getRange(const size_t &size);

/** Specializations of for getRange each supported
 *  dimensionality
 */
template <>
inline cl::sycl::range<1> getRange<1>(const size_t &size) {
  return cl::sycl::range<1>(size);
}
template <>
inline cl::sycl::range<2> getRange<2>(const size_t &size) {
  return cl::sycl::range<2>(size, size);
}
template <>
inline cl::sycl::range<3> getRange<3>(const size_t &size) {
  return cl::sycl::range<3>(size, size, size);
}

/**
 * @brief Helper function to check if all devices support online compiler.
 */
inline bool is_compiler_available(
    const cl::sycl::vector_class<cl::sycl::device>& deviceList) {
  bool compiler_available = true;
  for (const auto& device : deviceList) {
    if (!device.get_info<cl::sycl::info::device::is_compiler_available>()) {
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
    const cl::sycl::vector_class<cl::sycl::device>& deviceList) {
  bool linker_available = true;
  for (const auto& device : deviceList) {
    if (!device.get_info<cl::sycl::info::device::is_linker_available>()) {
      linker_available = false;
      break;
    }
  }
  return linker_available;
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

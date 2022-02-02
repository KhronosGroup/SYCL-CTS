/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Common functions for legacy multi_ptr API tests
//
*******************************************************************************/

#ifndef SYCL_CTS_TEST_MULTI_PTR_MULTI_PTR_LEGACY_API_COMMON_H
#define SYCL_CTS_TEST_MULTI_PTR_MULTI_PTR_LEGACY_API_COMMON_H

#include "../common/common.h"
#include "multi_ptr_common.h"
#include <algorithm>
#include <map>
#include <string>
#include <type_traits>

namespace multi_ptr_legacy_api_common {
using namespace sycl_cts;
using namespace multi_ptr_common;

template <typename T, typename U>
class kernel0;

template <typename From, typename To>
struct cast_keep_const {
  using type = To;
};
template <typename From, typename To>
struct cast_keep_const<const From, To> {
  using type = const To;
};

/** @brief Provides reference data to access through multi_ptr
 */
template <typename dataT, typename storageT = dataT>
struct reference {
  /** @brief Provides reference data size
   */
  static constexpr int size = 64;

  /** @brief Provides reference data values
   */
  static constexpr dataT value(int index) { return dataT(index + 1); }

  /** @brief Verifies the data pointed by multi_ptr, multi_ptr::pointer or raw
   *         pointer instance is equal to the reference data values
   */
  template <typename pointerT>
  static bool is_data_equal(pointerT &&ptr) {
    bool result = true;

    const auto rawStoragePtr = static_cast<storageT *>(ptr);
    const auto rawDataPtr = static_cast<dataT *>(rawStoragePtr);

    for (int i = 0; i < size; ++i) {
      result &= rawDataPtr[i] == value(i);
    }
    return result;
  }
};

/** @brief Provides enum for all checks with result verification
 */
enum class check_id : size_t {
  copy_assignment = 0,
  move_assignment,
  pointer_assignment,
  get_method,
  prefetch_method,
  raw_pointer_conversion,
  access_operators,
  arithmetic_operators,
  make_ptr_method,
  N_CHECKS  // should be last in enum
};

/** @brief Provides verification methods for generic types
 */
template <typename T, typename U>
class generic_check_helper;

/** @brief Provides verification methods for `void` types
 */
template <typename T, typename U>
class void_check_helper;

/** @brief Helper for generic check_helper alias specialization
 */
template <typename T, typename U>
struct check_helper_impl {
  using type = generic_check_helper<T, U>;
};
/** @brief Helper for `void` check_helper alias specialization
 */
template <typename T>
struct check_helper_impl<T, void> {
  using type = void_check_helper<T, void>;
};
/** @brief Helper for `const void` check_helper alias specialization
 */
template <typename T>
struct check_helper_impl<T, const void> {
  using type = void_check_helper<T, const void>;
};

/** @brief Alias for specific helper depending on types
 */
template <typename T, typename U = T>
using check_helper = typename check_helper_impl<T, U>::type;

/** @brief Provides common methods for any types using CRTP for static
 *         polymorphism
 */
template <typename T, typename U,
          template <typename, typename> class specificHelperT>
class core_check_helper {
 public:
  using multiPtrGlobal =
      multi_ptr_legacy<U, sycl::access::address_space::global_space>;
  using multiPtrConstant =
      multi_ptr_legacy<U, sycl::access::address_space::constant_space>;
  using multiPtrLocal =
      multi_ptr_legacy<U, sycl::access::address_space::local_space>;
  using multiPtrPrivate =
      multi_ptr_legacy<U, sycl::access::address_space::private_space>;

  template <sycl::access::address_space Space>
  bool pointer_assignment(multi_ptr_legacy<U, Space> multiPtr,
                          T *elementTypePtr) const {
    auto elementT = static_cast<U *>(elementTypePtr);

    // Check assigning nullptr_t
    multiPtr = nullptr;
    if (static_cast<U *>(multiPtr.get()) != nullptr) return false;

    // Check assigning ElementType*
    multiPtr = elementT;
    if (static_cast<U *>(multiPtr.get()) != elementT) return false;

    // Prepare for the next test
    auto pointerT = multiPtr.get();
    multiPtr = nullptr;

    // Check assigning pointer_t
    multiPtr = pointerT;
    if (static_cast<U *>(multiPtr.get()) != elementT) return false;

    return true;
  }

  void const_conversion_operators(multiPtrGlobal globalMultiPtr,
                                  multiPtrConstant constantMultiPtr,
                                  multiPtrLocal localMultiPtr,
                                  multiPtrPrivate privateMultiPtr) const {
    // Convert from U to const U
    auto cgPtr = static_cast<sycl::global_ptr<const U>>(globalMultiPtr);
    auto ccPtr = static_cast<sycl::constant_ptr<const U>>(constantMultiPtr);
    auto clPtr = static_cast<sycl::local_ptr<const U>>(localMultiPtr);
    auto cpPtr = static_cast<sycl::private_ptr<const U>>(privateMultiPtr);

    ASSERT_RETURN_TYPE(sycl::global_ptr<const U>, cgPtr,
                       "sycl::multi_ptr operator sycl::multi_ptr<const U, "
                       "sycl::access::address_space::global_space>()");
    ASSERT_RETURN_TYPE(sycl::constant_ptr<const U>, ccPtr,
                       "sycl::multi_ptr operator sycl::multi_ptr<const U, "
                       "sycl::access::address_space::constant_space>()");
    ASSERT_RETURN_TYPE(sycl::local_ptr<const U>, clPtr,
                       "sycl::multi_ptr operator sycl::multi_ptr<const U, "
                       "sycl::access::address_space::local_space>()");
    ASSERT_RETURN_TYPE(sycl::private_ptr<const U>, cpPtr,
                       "sycl::multi_ptr operator sycl::multi_ptr<const U, "
                       "sycl::access::address_space::private_space>()");
  }

  void arrow_operators(multiPtrGlobal globalMultiPtr,
                       multiPtrConstant constantMultiPtr,
                       multiPtrLocal localMultiPtr,
                       multiPtrPrivate privateMultiPtr) const {
    // primitives do not have any members
    return;
  }

  template <sycl::access::address_space Space>
  void relational_operators(multi_ptr_legacy<U, Space> multiPtr) const {
#define TEST_RELATION_OPERATOR_TEMPLATE(OP, LHS, RHS, LHS_TY_STR, RHS_TY_STR) \
  {                                                                           \
    auto res = LHS OP RHS;                                                    \
                                                                              \
    ASSERT_RETURN_TYPE(                                                       \
        bool, res, "sycl::operator" #OP "(" LHS_TY_STR ", " RHS_TY_STR ")");  \
  }

#define TEST_RELATION_OPERATOR(LHS, RHS, LHS_TY_STR, RHS_TY_STR)        \
  TEST_RELATION_OPERATOR_TEMPLATE(==, LHS, RHS, LHS_TY_STR, RHS_TY_STR) \
  TEST_RELATION_OPERATOR_TEMPLATE(!=, LHS, RHS, LHS_TY_STR, RHS_TY_STR) \
  TEST_RELATION_OPERATOR_TEMPLATE(<, LHS, RHS, LHS_TY_STR, RHS_TY_STR)  \
  TEST_RELATION_OPERATOR_TEMPLATE(>, LHS, RHS, LHS_TY_STR, RHS_TY_STR)  \
  TEST_RELATION_OPERATOR_TEMPLATE(<=, LHS, RHS, LHS_TY_STR, RHS_TY_STR) \
  TEST_RELATION_OPERATOR_TEMPLATE(>=, LHS, RHS, LHS_TY_STR, RHS_TY_STR)

    TEST_RELATION_OPERATOR(multiPtr, multiPtr, "sycl::multi_ptr",
                           "sycl::multi_ptr");
    TEST_RELATION_OPERATOR(nullptr, multiPtr, "std::nullptr_t",
                           "sycl::multi_ptr");
    TEST_RELATION_OPERATOR(multiPtr, nullptr, "sycl::multi_ptr",
                           "std::nullptr_t");
#undef TEST_RELATION_OPERATOR_TEMPLATE
#undef TEST_RELATION_OPERATOR
  }

  template <sycl::access::address_space Space>
  void member_types(multi_ptr_legacy<U, Space> multiPtr) const {
    using multi_ptr_t = multi_ptr_legacy<U, Space>;
    static_assert(std::is_same<multi_ptr_t, multi_ptr_legacy<U, Space>>::value,
                  "Invalid multi_ptr type");

    using difference_type = typename multi_ptr_t::difference_type;
    static_assert(std::is_same<difference_type, std::ptrdiff_t>::value,
                  "Difference type is wrong");

    using element_type = typename multi_ptr_t::element_type;
    static_assert(!std::is_same<element_type, std::nullptr_t>::value,
                  "Invalid element_type type");

    using pointer_t = typename multi_ptr_t::pointer_t;
    static_assert(std::is_pointer<pointer_t>::value,
                  "pointer_t is not a pointer type");
    static_assert(!std::is_same<pointer_t, std::nullptr_t>::value,
                  "Invalid pointer_t type");

    using const_pointer_t = typename multi_ptr_t::const_pointer_t;
    static_assert(std::is_pointer<const_pointer_t>::value,
                  "const_pointer_t is not a pointer type");
    static_assert(!std::is_same<const_pointer_t, std::nullptr_t>::value,
                  "Invalid const_pointer_t type");

    specificHelperT<T, U>::reference_member_types(multiPtr);
  }

  template <sycl::access::address_space Space>
  void address_space_member(multi_ptr_legacy<U, Space> multiPtr) const {
    static constexpr sycl::access::address_space addressSpace =
        multi_ptr_legacy<U, Space>::address_space;
    static_assert(addressSpace == Space, "Wrong address space");
  }
};

template <typename T, typename U>
class generic_check_helper
    : public core_check_helper<T, U, generic_check_helper> {
 public:
  using core_type = core_check_helper<T, U, generic_check_helper>;
  using multiPtrGlobal = typename core_type::multiPtrGlobal;
  using multiPtrConstant = typename core_type::multiPtrConstant;
  using multiPtrLocal = typename core_type::multiPtrLocal;
  using multiPtrPrivate = typename core_type::multiPtrPrivate;

  using data_void_t = typename cast_keep_const<T, void>::type;
  using reference_t = reference<T, U>;

  static void conversion_operators(multiPtrGlobal globalMultiPtr,
                                   multiPtrConstant constantMultiPtr,
                                   multiPtrLocal localMultiPtr,
                                   multiPtrPrivate privateMultiPtr) {
    using voidMultiPtrGlobal =
        multi_ptr_legacy<data_void_t,
                         sycl::access::address_space::global_space>;
    using voidMultiPtrConstant =
        multi_ptr_legacy<data_void_t,
                         sycl::access::address_space::constant_space>;
    using voidMultiPtrLocal =
        multi_ptr_legacy<data_void_t, sycl::access::address_space::local_space>;
    using voidMultiPtrPrivate =
        multi_ptr_legacy<data_void_t,
                         sycl::access::address_space::private_space>;

    // Convert from U to void
    auto gPtr = static_cast<voidMultiPtrGlobal>(globalMultiPtr);
    auto cPtr = static_cast<voidMultiPtrConstant>(constantMultiPtr);
    auto lPtr = static_cast<voidMultiPtrLocal>(localMultiPtr);
    auto pPtr = static_cast<voidMultiPtrPrivate>(privateMultiPtr);

    ASSERT_RETURN_TYPE(voidMultiPtrGlobal, gPtr,
                       "sycl::multi_ptr operator sycl::multi_ptr<void, "
                       "sycl::access::address_space::global_space>()");
    ASSERT_RETURN_TYPE(voidMultiPtrConstant, cPtr,
                       "sycl::multi_ptr operator sycl::multi_ptr<void, "
                       "sycl::access::address_space::constant_space>()");
    ASSERT_RETURN_TYPE(voidMultiPtrLocal, lPtr,
                       "sycl::multi_ptr operator sycl::multi_ptr<void, "
                       "sycl::access::address_space::local_space>()");
    ASSERT_RETURN_TYPE(voidMultiPtrPrivate, pPtr,
                       "sycl::multi_ptr operator sycl::multi_ptr<void, "
                       "sycl::access::address_space::private_space>()");
  }

  static bool access_operators(multiPtrGlobal globalMultiPtr,
                               multiPtrConstant constantMultiPtr,
                               multiPtrLocal localMultiPtr,
                               multiPtrPrivate privateMultiPtr) {
    {
      bool result = true;

      U globalElem = (*globalMultiPtr);
      U constantElem = (*constantMultiPtr);
      U localElem = (*localMultiPtr);
      U privateElem = (*privateMultiPtr);

      ASSERT_RETURN_TYPE(U, globalElem, "sycl::multi_ptr operator*()");
      ASSERT_RETURN_TYPE(U, constantElem, "sycl::multi_ptr operator*()");
      ASSERT_RETURN_TYPE(U, localElem, "sycl::multi_ptr operator*()");
      ASSERT_RETURN_TYPE(U, privateElem, "sycl::multi_ptr operator*()");

      // Verify access result
      const auto expected = reference_t::value(0);
      result &= globalElem == expected;
      result &= constantElem == expected;
      result &= localElem == expected;
      result &= privateElem == expected;

      // Verify underlying data for multi_ptr is not modified
      result &= reference_t::is_data_equal(globalMultiPtr);
      result &= reference_t::is_data_equal(constantMultiPtr);
      result &= reference_t::is_data_equal(localMultiPtr);
      result &= reference_t::is_data_equal(privateMultiPtr);

      return result;
    }
  }

  static bool prefetch_operation(multiPtrGlobal globalMultiPtr) {
    bool result = true;

    // Verify underlying data for multi_ptr is not modified by prefetch
    globalMultiPtr.prefetch(0);
    result &= reference_t::is_data_equal(globalMultiPtr);

    globalMultiPtr.prefetch(reference_t::size / 2);
    result &= reference_t::is_data_equal(globalMultiPtr);

    return result;
  }

  template <typename multiPtrT>
  static bool arithmetic_operators(multiPtrT multiPtr) {
    bool result = true;
    auto value = multiPtr.get();
    auto expected = value;

    std::ptrdiff_t diff = reference_t::size * 2;

    {
      expected = value++;

      auto retval = multiPtr++;
      ASSERT_RETURN_TYPE(multiPtrT, retval, "sycl::multi_ptr operator++(int)");
      result &= retval.get() == expected;
      result &= multiPtr.get() == value;
    }
    {
      expected = ++value;

      auto retval = ++multiPtr;
      ASSERT_RETURN_TYPE(multiPtrT, retval, "sycl::multi_ptr operator++()");
      result &= retval.get() == expected;
      result &= multiPtr.get() == value;
    }
    {
      expected = value--;

      auto retval = multiPtr--;
      ASSERT_RETURN_TYPE(multiPtrT, retval, "sycl::multi_ptr operator--(int)");
      result &= retval.get() == expected;
      result &= multiPtr.get() == value;
    }
    {
      expected = --value;

      auto retval = --multiPtr;
      ASSERT_RETURN_TYPE(multiPtrT, retval, "sycl::multi_ptr operator--()");
      result &= retval.get() == expected;
      result &= multiPtr.get() == value;
    }
    {
      expected = value + diff;

      auto retval = multiPtr + diff;
      ASSERT_RETURN_TYPE(multiPtrT, retval,
                         "sycl::multi_ptr operator+(std::ptrdiff_t r)");
      result &= retval.get() == expected;
      result &= multiPtr.get() == value;
    }
    {
      expected = value - diff;

      auto retval = multiPtr - diff;
      ASSERT_RETURN_TYPE(multiPtrT, retval,
                         "sycl::multi_ptr operator-(std::ptrdiff_t r)");
      result &= retval.get() == expected;
      result &= multiPtr.get() == value;
    }
    {
      expected = value += diff;

      auto retval = multiPtr += diff;
      ASSERT_RETURN_TYPE(multiPtrT, retval,
                         "sycl::multi_ptr operator+=(std::ptrdiff_t r)");
      result &= retval.get() == expected;
      result &= multiPtr.get() == value;
    }
    {
      expected = value -= diff;

      auto retval = multiPtr -= diff;
      ASSERT_RETURN_TYPE(multiPtrT, retval,
                         "sycl::multi_ptr operator-=(std::ptrdiff_t r)");
      result &= retval.get() == expected;
      result &= multiPtr.get() == value;
    }
    return result;
  }

  static bool arithmetic_operators(multiPtrGlobal globalMultiPtr,
                                   multiPtrConstant constantMultiPtr,
                                   multiPtrLocal localMultiPtr,
                                   multiPtrPrivate privateMultiPtr) {
    return arithmetic_operators(globalMultiPtr) &&
           arithmetic_operators(constantMultiPtr) &&
           arithmetic_operators(localMultiPtr) &&
           arithmetic_operators(privateMultiPtr);
  }

  template <sycl::access::address_space Space>
  static void reference_member_types(multi_ptr_legacy<U, Space> multiPtr) {
    using multi_ptr_t = multi_ptr_legacy<U, Space>;
    static_assert(std::is_same<multi_ptr_t, multi_ptr_legacy<U, Space>>::value,
                  "Invalid multi_ptr type");

    using reference_t = typename multi_ptr_t::reference_t;
    static_assert(std::is_reference<reference_t>::value,
                  "reference_t is not a reference type");
    static_assert(!std::is_same<reference_t, std::nullptr_t>::value,
                  "Invalid reference_t type");

    using const_reference_t = typename multi_ptr_t::const_reference_t;
    static_assert(std::is_reference<const_reference_t>::value,
                  "const_reference_t is not a reference type");
    static_assert(!std::is_same<const_reference_t, std::nullptr_t>::value,
                  "Invalid const_reference_t type");
  }
};

/** @brief Provide methods for 'void' and 'const void' types
 */
template <typename T, typename VoidT>
class void_check_helper
    : public core_check_helper<T, VoidT, void_check_helper> {
 public:
  using core_type = core_check_helper<T, VoidT, void_check_helper>;
  using multiPtrGlobal = typename core_type::multiPtrGlobal;
  using multiPtrConstant = typename core_type::multiPtrConstant;
  using multiPtrLocal = typename core_type::multiPtrLocal;
  using multiPtrPrivate = typename core_type::multiPtrPrivate;

  static void conversion_operators(multiPtrGlobal globalMultiPtr,
                                   multiPtrConstant constantMultiPtr,
                                   multiPtrLocal localMultiPtr,
                                   multiPtrPrivate privateMultiPtr) {
    using float_t = typename cast_keep_const<T, float>::type;

    using floatMultiPtrGlobal =
        multi_ptr_legacy<float_t, sycl::access::address_space::global_space>;
    using floatMultiPtrConstant =
        multi_ptr_legacy<float_t, sycl::access::address_space::constant_space>;
    using floatMultiPtrLocal =
        multi_ptr_legacy<float_t, sycl::access::address_space::local_space>;
    using floatMultiPtrPrivate =
        multi_ptr_legacy<float_t, sycl::access::address_space::private_space>;

    auto gPtr = static_cast<floatMultiPtrGlobal>(globalMultiPtr);
    auto cPtr = static_cast<floatMultiPtrConstant>(constantMultiPtr);
    auto lPtr = static_cast<floatMultiPtrLocal>(localMultiPtr);
    auto pPtr = static_cast<floatMultiPtrPrivate>(privateMultiPtr);

    ASSERT_RETURN_TYPE(floatMultiPtrGlobal, gPtr,
                       "sycl::multi_ptr operator sycl::multi_ptr<float, "
                       "sycl::access::address_space::global_space>()");
    ASSERT_RETURN_TYPE(floatMultiPtrConstant, cPtr,
                       "sycl::multi_ptr operator sycl::multi_ptr<float, "
                       "sycl::access::address_space::constant_space>()");
    ASSERT_RETURN_TYPE(floatMultiPtrLocal, lPtr,
                       "sycl::multi_ptr operator sycl::multi_ptr<float, "
                       "sycl::access::address_space::local_space>()");
    ASSERT_RETURN_TYPE(floatMultiPtrPrivate, pPtr,
                       "sycl::multi_ptr operator sycl::multi_ptr<float, "
                       "sycl::access::address_space::private_space>()");
  }

  static bool prefetch_operation(multiPtrGlobal) {
    // void type does not support prefeth operation
    return true;
  }

  static bool access_operators(multiPtrGlobal, multiPtrConstant, multiPtrLocal,
                               multiPtrPrivate) {
    // void type does not support access operators
    return true;
  }

  static bool arithmetic_operators(multiPtrGlobal, multiPtrConstant,
                                   multiPtrLocal, multiPtrPrivate) {
    // void type does not support arithmetic operators
    return true;
  }

  template <sycl::access::address_space Space>
  static void reference_member_types(multi_ptr_legacy<VoidT, Space>) {
    // multi_ptr<VoidT> does not have any reference member types
  }
};

template <typename T, typename U = T>
class pointer_apis {
 private:
  using data_t = typename std::remove_const<T>::type;
  using reference_t = reference<T, U>;

  using multiPtrGlobal =
      multi_ptr_legacy<U, sycl::access::address_space::global_space>;
  using multiPtrConstant =
      multi_ptr_legacy<U, sycl::access::address_space::constant_space>;
  using multiPtrLocal =
      multi_ptr_legacy<U, sycl::access::address_space::local_space>;
  using multiPtrPrivate =
      multi_ptr_legacy<U, sycl::access::address_space::private_space>;

  /** @brief Provides error message for any check with result verification
   */
  std::string construct_error_message(check_id id, std::string dataTypeName,
                                      std::string storageTypeName) {
    static const std::map<check_id, const char *> names {
      {check_id::copy_assignment, "copy assignment"},
          {check_id::move_assignment, "move assignment"},
          {check_id::pointer_assignment, "pointer assignment"},
          {check_id::get_method, "get() method"},
          {check_id::prefetch_method, "prefetch() method"},
          {check_id::raw_pointer_conversion, "raw pointer conversion"},
          {check_id::access_operators, "access operators"},
          {check_id::arithmetic_operators, "arithmetic operators"},
          {check_id::make_ptr_method, "make_ptr() method"}};

    constexpr bool isConstDataType =
      std::is_same_v<T, typename std::remove_const<T>::type>;
    constexpr bool isConstStorageType =
      std::is_same_v<U, typename std::remove_const<U>::type>;

    if constexpr (isConstDataType) {
      dataTypeName = "const " + dataTypeName;
    }
    if constexpr (isConstStorageType) {
      storageTypeName = "const " + storageTypeName;
    }

    std::string result{names.at(id)};
    result += " with data and storage types <";
    result += dataTypeName + ", " + storageTypeName + "> failed";

    return result;
  }

 public:
  void operator()(util::logger &log, sycl::queue &queue,
                  const std::string& dataTypeName) {
    return operator() (log, queue, dataTypeName, dataTypeName);
  }
  void operator()(util::logger &log, sycl::queue &queue,
                  const std::string& dataTypeName,
                  const std::string& storageTypeName) {
    constexpr auto nChecks = to_integral(check_id::N_CHECKS);
    const auto size = reference_t::size;
    sycl::range<1> range(size);
    sycl::range<1> resRange(nChecks);
    std::unique_ptr<data_t[]> data(new data_t[size]);

    bool pass[nChecks];
    std::fill_n(pass, nChecks, false);

    for (int i = 0; i < size; ++i) {
      data[i] = reference_t::value(i);
    }

    {
      sycl::buffer<bool, 1> resBuff(pass, resRange);
      sycl::buffer<T, 1> buffer(data.get(), range);

      queue.submit([&](sycl::handler &handler) {
        auto resAcc =
              resBuff.get_access<sycl::access_mode::read_write>(handler);
        sycl::accessor<T, 1, sycl::access_mode::read_write,
                           sycl::target::device>
            globalAccessor(buffer, handler);
        sycl::accessor<T, 1, sycl::access_mode::read,
                           sycl::target::constant_buffer>
            constantAccessor(buffer, handler);
        sycl::accessor<T, 1, sycl::access_mode::read_write,
                           sycl::target::local>
            localAccessor(size, handler);

        handler.single_task<class kernel0<T, U>>(
              [resAcc, globalAccessor, constantAccessor, localAccessor]() {
          check_helper<T, U> checker;

          data_t privateData[size];
          data_t *localData = const_cast<data_t*>(&localAccessor[0]);

          for (int i = 0; i < size; ++i) {
            privateData[i] = reference_t::value(i);
            localData[i] = reference_t::value(i);
          }

          // Reference pointer values to check against
          const auto expectedConstantPtr =
              const_cast<U *>(static_cast<const U *>(&constantAccessor[0]));
          const auto expectedGlobalPtr = static_cast<U *>(&globalAccessor[0]);
          const auto expectedLocalPtr = static_cast<U *>(&localAccessor[0]);
          const auto expectedPrivatePtr = static_cast<U *>(privateData);

          /** check multi_ptr aliases
           */
          {
            static_assert(
              std::is_same<multiPtrGlobal, global_ptr_legacy<U>>::value,
              "Invalid global_ptr type");
            static_assert(
              std::is_same<multiPtrConstant, constant_ptr_legacy<U>>::value,
              "Invalid constant_ptr type");
            static_assert(
              std::is_same<multiPtrLocal, local_ptr_legacy<U>>::value,
              "Invalid local_ptr type");
            static_assert(
              std::is_same<multiPtrPrivate, private_ptr_legacy<U>>::value,
              "Invalid private_ptr type");
          }

          /** check member types
           */
          {
            // construct a set of multi_ptr
            global_ptr_legacy<U> globalPtr(expectedGlobalPtr);
            constant_ptr_legacy<U> constantPtr(expectedConstantPtr);
            local_ptr_legacy<U> localPtr(expectedLocalPtr);
            private_ptr_legacy<U> privatePtr(expectedPrivatePtr);

            multiPtrGlobal globalMultiPtr(globalPtr);
            multiPtrConstant constantMultiPtr(constantPtr);
            multiPtrLocal localMultiPtr(localPtr);
            multiPtrPrivate privateMultiPtr(privatePtr);

            checker.member_types(globalMultiPtr);
            checker.member_types(constantMultiPtr);
            checker.member_types(localMultiPtr);
            checker.member_types(privateMultiPtr);
          }

          /** check address_space member
           */
          {
            // construct a set of multi_ptr
            sycl::global_ptr<U> globalPtr(expectedGlobalPtr);
            sycl::constant_ptr<U> constantPtr(expectedConstantPtr);
            sycl::local_ptr<U> localPtr(expectedLocalPtr);
            sycl::private_ptr<U> privatePtr(expectedPrivatePtr);

            multiPtrGlobal globalMultiPtr(globalPtr);
            multiPtrConstant constantMultiPtr(constantPtr);
            multiPtrLocal localMultiPtr(localPtr);
            multiPtrPrivate privateMultiPtr(privatePtr);

            checker.address_space_member(globalMultiPtr);
            checker.address_space_member(constantMultiPtr);
            checker.address_space_member(localMultiPtr);
            checker.address_space_member(privateMultiPtr);
          }

          /** check copy assignment operators
           */
          {
            // construct two sets of multi_ptr
            global_ptr_legacy<U> globalPtrA(expectedGlobalPtr);
            constant_ptr_legacy<U> constantPtrA(expectedConstantPtr);
            local_ptr_legacy<U> localPtrA(expectedLocalPtr);
            private_ptr_legacy<U> privatePtrA(expectedPrivatePtr);

            multiPtrGlobal globalMultiPtrA(globalPtrA);
            multiPtrConstant constantMultiPtrA(constantPtrA);
            multiPtrLocal localMultiPtrA(localPtrA);
            multiPtrPrivate privateMultiPtrA(privatePtrA);

            multiPtrGlobal globalMultiPtrB;
            multiPtrConstant constantMultiPtrB;
            multiPtrLocal localMultiPtrB;
            multiPtrPrivate privateMultiPtrB;

            // check copy assignment operators
            globalMultiPtrB = globalMultiPtrA;
            constantMultiPtrB = constantMultiPtrA;
            localMultiPtrB = localMultiPtrA;
            privateMultiPtrB = privateMultiPtrA;

            bool result = true;
            result &= globalMultiPtrB == expectedGlobalPtr;
            result &= constantMultiPtrB == expectedConstantPtr;
            result &= localMultiPtrB == expectedLocalPtr;
            result &= privateMultiPtrB == expectedPrivatePtr;

            result &= reference_t::is_data_equal(globalMultiPtrB);
            result &= reference_t::is_data_equal(constantMultiPtrB);
            result &= reference_t::is_data_equal(localMultiPtrB);
            result &= reference_t::is_data_equal(privateMultiPtrB);

            resAcc[to_integral(check_id::copy_assignment)] = result;
          }

          /** check move assignment operators
           */
          {
            // construct two sets of multi_ptr
            global_ptr_legacy<U> globalPtrA(expectedGlobalPtr);
            constant_ptr_legacy<U> constantPtrA(expectedConstantPtr);
            local_ptr_legacy<U> localPtrA(expectedLocalPtr);
            private_ptr_legacy<U> privatePtrA(expectedPrivatePtr);

            multiPtrGlobal globalMultiPtrA(globalPtrA);
            multiPtrConstant constantMultiPtrA(constantPtrA);
            multiPtrLocal localMultiPtrA(localPtrA);
            multiPtrPrivate privateMultiPtrA(privatePtrA);

            multiPtrGlobal globalMultiPtrB;
            multiPtrConstant constantMultiPtrB;
            multiPtrLocal localMultiPtrB;
            multiPtrPrivate privateMultiPtrB;

            // check move assignment operators
            globalMultiPtrB = std::move(globalMultiPtrA);
            constantMultiPtrB = std::move(constantMultiPtrA);
            localMultiPtrB = std::move(localMultiPtrA);
            privateMultiPtrB = std::move(privateMultiPtrA);

            bool result = true;
            result &= globalMultiPtrB == expectedGlobalPtr;
            result &= constantMultiPtrB == expectedConstantPtr;
            result &= localMultiPtrB == expectedLocalPtr;
            result &= privateMultiPtrB == expectedPrivatePtr;

            result &= reference_t::is_data_equal(globalMultiPtrB);
            result &= reference_t::is_data_equal(constantMultiPtrB);
            result &= reference_t::is_data_equal(localMultiPtrB);
            result &= reference_t::is_data_equal(privateMultiPtrB);

            resAcc[to_integral(check_id::move_assignment)] = result;
          }

          /** check assigning to multi_ptr
           */
          {
            // construct a set of multi_ptr
            global_ptr_legacy<U> globalPtr(expectedGlobalPtr);
            constant_ptr_legacy<U> constantPtr(expectedConstantPtr);
            local_ptr_legacy<U> localPtr(expectedLocalPtr);
            private_ptr_legacy<U> privatePtr(expectedPrivatePtr);

            multiPtrGlobal globalMultiPtr(globalPtr);
            multiPtrConstant constantMultiPtr(constantPtr);
            multiPtrLocal localMultiPtr(localPtr);
            multiPtrPrivate privateMultiPtr(privatePtr);

            bool result = true;
            result &= globalMultiPtr == expectedGlobalPtr;
            result &= constantMultiPtr == expectedConstantPtr;
            result &= localMultiPtr == expectedLocalPtr;
            result &= privateMultiPtr == expectedPrivatePtr;

            result &= reference_t::is_data_equal(globalMultiPtr);
            result &= reference_t::is_data_equal(constantMultiPtr);
            result &= reference_t::is_data_equal(localMultiPtr);
            result &= reference_t::is_data_equal(privateMultiPtr);

            resAcc[to_integral(check_id::pointer_assignment)] = result;
          }

          /** check get() methods
           */
          {
            // construct a set of multi_ptr
            global_ptr_legacy<U> globalPtr(expectedGlobalPtr);
            constant_ptr_legacy<U> constantPtr(expectedConstantPtr);
            local_ptr_legacy<U> localPtr(expectedLocalPtr);
            private_ptr_legacy<U> privatePtr(expectedPrivatePtr);

            multiPtrGlobal globalMultiPtr(globalPtr);
            multiPtrConstant constantMultiPtr(constantPtr);
            multiPtrLocal localMultiPtr(localPtr);
            multiPtrPrivate privateMultiPtr(privatePtr);

            auto gPtr = globalMultiPtr.get();
            auto cPtr = constantMultiPtr.get();
            auto lPtr = localMultiPtr.get();
            auto pPtr = privateMultiPtr.get();

            ASSERT_RETURN_TYPE(typename global_ptr_legacy<U>::pointer_t,
                               gPtr, "sycl::multi_ptr::get()");
            ASSERT_RETURN_TYPE(typename constant_ptr_legacy<U>::pointer_t,
                               cPtr, "sycl::multi_ptr::get()");
            ASSERT_RETURN_TYPE(typename local_ptr_legacy<U>::pointer_t, lPtr,
                               "sycl::multi_ptr::get()");
            ASSERT_RETURN_TYPE(typename private_ptr_legacy<U>::pointer_t,
                               pPtr, "sycl::multi_ptr::get()");

            bool result = true;
            result &= gPtr == expectedGlobalPtr;
            result &= cPtr == expectedConstantPtr;
            result &= lPtr == expectedLocalPtr;
            result &= pPtr == expectedPrivatePtr;

            result &= reference_t::is_data_equal(gPtr);
            result &= reference_t::is_data_equal(cPtr);
            result &= reference_t::is_data_equal(lPtr);
            result &= reference_t::is_data_equal(pPtr);

            resAcc[to_integral(check_id::get_method)] = result;
          }

          /** check prefetch() method
           */
          {
            // construct a global multi_ptr
            global_ptr_legacy<U> globalPtr(expectedGlobalPtr);
            multiPtrGlobal globalMultiPtr(globalPtr);

            resAcc[to_integral(check_id::prefetch_method)] =
                checker.prefetch_operation(globalMultiPtr);
          }

          /** check implicit conversion to a raw pointer
           */
          {
            // construct a set of multi_ptr
            global_ptr_legacy<U> globalPtr(expectedGlobalPtr);
            constant_ptr_legacy<U> constantPtr(expectedConstantPtr);
            local_ptr_legacy<U> localPtr(expectedLocalPtr);
            private_ptr_legacy<U> privatePtr(expectedPrivatePtr);

            multiPtrGlobal globalMultiPtr(globalPtr);
            multiPtrConstant constantMultiPtr(constantPtr);
            multiPtrLocal localMultiPtr(localPtr);
            multiPtrPrivate privateMultiPtr(privatePtr);

            U *gPtr = globalMultiPtr;
            U *cPtr = constantMultiPtr;
            U *lPtr = localMultiPtr;
            U *pPtr = privateMultiPtr;

            bool result = true;
            result &= gPtr == expectedGlobalPtr;
            result &= cPtr == expectedConstantPtr;
            result &= lPtr == expectedLocalPtr;
            result &= pPtr == expectedPrivatePtr;

            result &= reference_t::is_data_equal(gPtr);
            result &= reference_t::is_data_equal(cPtr);
            result &= reference_t::is_data_equal(lPtr);
            result &= reference_t::is_data_equal(pPtr);

            resAcc[to_integral(check_id::raw_pointer_conversion)] = result;
          }

          /** check multi_ptr conversion methods
           */
          {
            // construct a set of multi_ptr
            global_ptr_legacy<U> globalPtr(expectedGlobalPtr);
            constant_ptr_legacy<U> constantPtr(expectedConstantPtr);
            local_ptr_legacy<U> localPtr(expectedLocalPtr);
            private_ptr_legacy<U> privatePtr(expectedPrivatePtr);

            multiPtrGlobal globalMultiPtr(globalPtr);
            multiPtrConstant constantMultiPtr(constantPtr);
            multiPtrLocal localMultiPtr(localPtr);
            multiPtrPrivate privateMultiPtr(privatePtr);

            checker.conversion_operators(globalMultiPtr, constantMultiPtr,
                                         localMultiPtr, privateMultiPtr);
            checker.const_conversion_operators(globalMultiPtr, constantMultiPtr,
                                               localMultiPtr, privateMultiPtr);
          }

          /** check operator[int]() methods
           *  check operator*() methods
           */
          {
            // construct a set of multi_ptr
            global_ptr_legacy<U> globalPtr(expectedGlobalPtr);
            constant_ptr_legacy<U> constantPtr(expectedConstantPtr);
            local_ptr_legacy<U> localPtr(expectedLocalPtr);
            private_ptr_legacy<U> privatePtr(expectedPrivatePtr);

            multiPtrGlobal globalMultiPtr(globalPtr);
            multiPtrConstant constantMultiPtr(constantPtr);
            multiPtrLocal localMultiPtr(localPtr);
            multiPtrPrivate privateMultiPtr(privatePtr);

            resAcc[to_integral(check_id::access_operators)] =
                checker.access_operators(globalMultiPtr, constantMultiPtr,
                                         localMultiPtr, privateMultiPtr);
          }

          /** check operator->() methods
           */
          {
            // construct a set of multi_ptr
            global_ptr_legacy<U> globalPtr(expectedGlobalPtr);
            constant_ptr_legacy<U> constantPtr(expectedConstantPtr);
            local_ptr_legacy<U> localPtr(expectedLocalPtr);
            private_ptr_legacy<U> privatePtr(expectedPrivatePtr);

            multiPtrGlobal globalMultiPtr(globalPtr);
            multiPtrConstant constantMultiPtr(constantPtr);
            multiPtrLocal localMultiPtr(localPtr);
            multiPtrPrivate privateMultiPtr(privatePtr);

            checker.arrow_operators(globalMultiPtr, constantMultiPtr,
                                    localMultiPtr, privateMultiPtr);
          }

          /** check arithmetic operators
           */
          {
            // construct a set of multi_ptr
            global_ptr_legacy<U> globalPtr(expectedGlobalPtr);
            constant_ptr_legacy<U> constantPtr(expectedConstantPtr);
            local_ptr_legacy<U> localPtr(expectedLocalPtr);
            private_ptr_legacy<U> privatePtr(expectedPrivatePtr);

            multiPtrGlobal globalMultiPtr(globalPtr);
            multiPtrConstant constantMultiPtr(constantPtr);
            multiPtrLocal localMultiPtr(localPtr);
            multiPtrPrivate privateMultiPtr(privatePtr);

            resAcc[to_integral(check_id::arithmetic_operators)] =
                checker.arithmetic_operators(globalMultiPtr, constantMultiPtr,
                                             localMultiPtr, privateMultiPtr);
          }

          /** check make_ptr function
           */
          {
            using namespace sycl::access;

            multiPtrGlobal globalMultiPtr =
                sycl::make_ptr<U, address_space::global_space,
                               decorated::legacy>(
                    expectedGlobalPtr);
            multiPtrConstant constantMultiPtr =
                sycl::make_ptr<U, address_space::constant_space,
                               decorated::legacy>(
                    expectedConstantPtr);
            multiPtrLocal localMultiPtr =
                sycl::make_ptr<U, address_space::local_space,
                               decorated::legacy>(
                    expectedLocalPtr);
            multiPtrPrivate privateMultiPtr =
                sycl::make_ptr<U, address_space::private_space,
                               decorated::legacy>(
                    expectedPrivatePtr);

            bool result = true;

            result &= reference_t::is_data_equal(globalMultiPtr);
            result &= reference_t::is_data_equal(constantMultiPtr);
            result &= reference_t::is_data_equal(localMultiPtr);
            result &= reference_t::is_data_equal(privateMultiPtr);

            resAcc[to_integral(check_id::make_ptr_method)] = result;
          }

          /** check relation functions
           */
          {
            // construct a set of multi_ptr
            global_ptr_legacy<U> globalPtr(expectedGlobalPtr);
            constant_ptr_legacy<U> constantPtr(expectedConstantPtr);
            local_ptr_legacy<U> localPtr(expectedLocalPtr);
            private_ptr_legacy<U> privatePtr(expectedPrivatePtr);

            checker.relational_operators(globalPtr);
            checker.relational_operators(constantPtr);
            checker.relational_operators(localPtr);
            checker.relational_operators(privatePtr);
          }
        });
      });
    } //end of buffer scope

    /** check space field assignment operators
     */
    {
      // construct two sets of multi_ptr
      global_ptr_legacy<U> globalPtrA;
      constant_ptr_legacy<U> constantPtrA;
      local_ptr_legacy<U> localPtrA;
      private_ptr_legacy<U> privatePtrA;

      static constexpr auto resG = decltype(globalPtrA)::address_space;
      static constexpr auto resC = decltype(constantPtrA)::address_space;
      static constexpr auto resL = decltype(localPtrA)::address_space;
      static constexpr auto resP = decltype(privatePtrA)::address_space;

      check_return_type<sycl::access::address_space>(
          log, resG, "sycl::multi_ptr::space");
      check_return_type<sycl::access::address_space>(
          log, resC, "sycl::multi_ptr::space");
      check_return_type<sycl::access::address_space>(
          log, resL, "sycl::multi_ptr::space");
      check_return_type<sycl::access::address_space>(
          log, resP, "sycl::multi_ptr::space");

      check_return_value<sycl::access::address_space>(
          log, resG, sycl::access::address_space::global_space,
          "sycl::multi_ptr::space");
      check_return_value<sycl::access::address_space>(
          log, resC, sycl::access::address_space::constant_space,
          "sycl::multi_ptr::space");
      check_return_value<sycl::access::address_space>(
          log, resL, sycl::access::address_space::local_space,
          "sycl::multi_ptr::space");
      check_return_value<sycl::access::address_space>(
          log, resP, sycl::access::address_space::private_space,
          "sycl::multi_ptr::space");
    }

    /** Report on failures
     */
    for (size_t i = 0; i < nChecks; ++i) {
      if (!pass[i]) {
        const auto errorDesc = construct_error_message(static_cast<check_id>(i),
                                                       dataTypeName,
                                                       storageTypeName);
        FAIL(log, errorDesc);
      }
    }
  }
};

template <typename T>
using check_pointer_api = check_pointer<pointer_apis, T>;

template <typename T>
using check_void_pointer_api = check_void_pointer<pointer_apis, T>;

} // namespace multi_ptr_legacy_api_common

#endif  // SYCL_CTS_TEST_MULTI_PTR_MULTI_PTR_LEGACY_API_COMMON_H

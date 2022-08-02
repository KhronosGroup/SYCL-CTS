/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2022 The Khronos Group Inc.
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

#include "../common/common.h"

#define TEST_NAME multi_ptr_apis

namespace TEST_NAME {
using namespace sycl_cts;

struct user_struct {
  float a;
  int b;
  char c;
};

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

template <typename T, typename U = T>
class check_helper {
 public:
  using multiPtrGlobal =
      cl::sycl::multi_ptr<U, cl::sycl::access::address_space::global_space>;
  using multiPtrConstant =
      cl::sycl::multi_ptr<U, cl::sycl::access::address_space::constant_space>;
  using multiPtrLocal =
      cl::sycl::multi_ptr<U, cl::sycl::access::address_space::local_space>;
  using multiPtrPrivate =
      cl::sycl::multi_ptr<U, cl::sycl::access::address_space::private_space>;

  using data_void_t = typename cast_keep_const<T, void>::type;

  template <cl::sycl::access::address_space Space>
  void pointer_assignment(cl::sycl::multi_ptr<U, Space> multiPtr,
                          T *elementTypePtr) const {
    // Check assigning pointer_t
    auto pointerT = multiPtr.get();
    multiPtr = pointerT;

    // Check assigning ElementType*
    multiPtr = static_cast<U *>(elementTypePtr);

    // Check assigning nullptr_t
    multiPtr = nullptr;
  }

  void conversion_operators(multiPtrGlobal globalMultiPtr,
                            multiPtrConstant constantMultiPtr,
                            multiPtrLocal localMultiPtr,
                            multiPtrPrivate privateMultiPtr) const {
    using voidMultiPtrGlobal =
        cl::sycl::multi_ptr<data_void_t,
                            cl::sycl::access::address_space::global_space>;
    using voidMultiPtrConstant =
        cl::sycl::multi_ptr<data_void_t,
                            cl::sycl::access::address_space::constant_space>;
    using voidMultiPtrLocal =
        cl::sycl::multi_ptr<data_void_t,
                            cl::sycl::access::address_space::local_space>;
    using voidMultiPtrPrivate =
        cl::sycl::multi_ptr<data_void_t,
                            cl::sycl::access::address_space::private_space>;

    // Convert from U to void
    auto gPtr = static_cast<voidMultiPtrGlobal>(globalMultiPtr);
    auto cPtr = static_cast<voidMultiPtrConstant>(constantMultiPtr);
    auto lPtr = static_cast<voidMultiPtrLocal>(localMultiPtr);
    auto pPtr = static_cast<voidMultiPtrPrivate>(privateMultiPtr);

    ASSERT_RETURN_TYPE(voidMultiPtrGlobal, gPtr,
                       "cl::sycl::multi_ptr operator cl::sycl::multi_ptr<void, "
                       "cl::sycl::access::address_space::global_space>()");
    ASSERT_RETURN_TYPE(voidMultiPtrConstant, cPtr,
                       "cl::sycl::multi_ptr operator cl::sycl::multi_ptr<void, "
                       "cl::sycl::access::address_space::constant_space>()");
    ASSERT_RETURN_TYPE(voidMultiPtrLocal, lPtr,
                       "cl::sycl::multi_ptr operator cl::sycl::multi_ptr<void, "
                       "cl::sycl::access::address_space::local_space>()");
    ASSERT_RETURN_TYPE(voidMultiPtrPrivate, pPtr,
                       "cl::sycl::multi_ptr operator cl::sycl::multi_ptr<void, "
                       "cl::sycl::access::address_space::private_space>()");
  }

  void const_conversion_operators(multiPtrGlobal globalMultiPtr,
                                  multiPtrConstant constantMultiPtr,
                                  multiPtrLocal localMultiPtr,
                                  multiPtrPrivate privateMultiPtr) const {
    // Convert from U to const U
    auto cgPtr = static_cast<cl::sycl::global_ptr<const U>>(globalMultiPtr);
    auto ccPtr = static_cast<cl::sycl::constant_ptr<const U>>(constantMultiPtr);
    auto clPtr = static_cast<cl::sycl::local_ptr<const U>>(localMultiPtr);
    auto cpPtr = static_cast<cl::sycl::private_ptr<const U>>(privateMultiPtr);

    ASSERT_RETURN_TYPE(
        cl::sycl::global_ptr<const U>, cgPtr,
        "cl::sycl::multi_ptr operator cl::sycl::multi_ptr<const U, "
        "cl::sycl::access::address_space::global_space>()");
    ASSERT_RETURN_TYPE(
        cl::sycl::constant_ptr<const U>, ccPtr,
        "cl::sycl::multi_ptr operator cl::sycl::multi_ptr<const U, "
        "cl::sycl::access::address_space::constant_space>()");
    ASSERT_RETURN_TYPE(
        cl::sycl::local_ptr<const U>, clPtr,
        "cl::sycl::multi_ptr operator cl::sycl::multi_ptr<const U, "
        "cl::sycl::access::address_space::local_space>()");
    ASSERT_RETURN_TYPE(
        cl::sycl::private_ptr<const U>, cpPtr,
        "cl::sycl::multi_ptr operator cl::sycl::multi_ptr<const U, "
        "cl::sycl::access::address_space::private_space>()");
  }

  void access_operators(multiPtrGlobal globalMultiPtr,
                        multiPtrConstant constantMultiPtr,
                        multiPtrLocal localMultiPtr,
                        multiPtrPrivate privateMultiPtr) const {
    {
      U globalElem = (*globalMultiPtr);
      U constantElem = (*constantMultiPtr);
      U localElem = (*localMultiPtr);
      U privateElem = (*privateMultiPtr);

      ASSERT_RETURN_TYPE(U, globalElem, "cl::sycl::multi_ptr operator*()");
      ASSERT_RETURN_TYPE(U, constantElem, "cl::sycl::multi_ptr operator*()");
      ASSERT_RETURN_TYPE(U, localElem, "cl::sycl::multi_ptr operator*()");
      ASSERT_RETURN_TYPE(U, privateElem, "cl::sycl::multi_ptr operator*()");
    }
  }

  void arrow_operators(multiPtrGlobal globalMultiPtr,
                       multiPtrConstant constantMultiPtr,
                       multiPtrLocal localMultiPtr,
                       multiPtrPrivate privateMultiPtr) const {
    // primitives do not have any members
    return;
  }

  void prefetch_operation(multiPtrGlobal globalMultiPtr) const {
    globalMultiPtr.prefetch(1);
  }

  void arithmetic_operators(multiPtrGlobal globalMultiPtr,
                            multiPtrConstant constantMultiPtr,
                            multiPtrLocal localMultiPtr,
                            multiPtrPrivate privateMultiPtr) const {
    std::ptrdiff_t diff = 10;

    // check all arithmetic operators for global multi_ptr
    auto globalMultiPtrIncPost = globalMultiPtr++;
    ASSERT_RETURN_TYPE(multiPtrGlobal, globalMultiPtrIncPost,
                       "cl::sycl::multi_ptr operator++(int)");
    auto globalMultiPtrIncPre = ++globalMultiPtr;
    ASSERT_RETURN_TYPE(multiPtrGlobal, globalMultiPtrIncPre,
                       "cl::sycl::multi_ptr operator++()");
    auto globalMultiPtrDecPost = globalMultiPtr--;
    ASSERT_RETURN_TYPE(multiPtrGlobal, globalMultiPtrDecPost,
                       "cl::sycl::multi_ptr operator--(int)");
    auto globalMultiPtrDecPre = --globalMultiPtr;
    ASSERT_RETURN_TYPE(multiPtrGlobal, globalMultiPtrDecPre,
                       "cl::sycl::multi_ptr operator--()");
    auto globalMultiPtrAdd = globalMultiPtr + diff;
    ASSERT_RETURN_TYPE(multiPtrGlobal, globalMultiPtrAdd,
                       "cl::sycl::multi_ptr operator+(std::ptrdiff_t r)");
    auto globalMultiPtrSub = globalMultiPtr - diff;
    ASSERT_RETURN_TYPE(multiPtrGlobal, globalMultiPtrSub,
                       "cl::sycl::multi_ptr operator-(std::ptrdiff_t r)");
    auto globalMultiPtrAddAssign = globalMultiPtr += diff;
    ASSERT_RETURN_TYPE(multiPtrGlobal, globalMultiPtrAddAssign,
                       "cl::sycl::multi_ptr operator+=(std::ptrdiff_t r)");
    auto globalMultiPtrSubAssign = globalMultiPtr -= diff;
    ASSERT_RETURN_TYPE(multiPtrGlobal, globalMultiPtrSubAssign,
                       "cl::sycl::multi_ptr operator-=(std::ptrdiff_t r)");

    // check all arithmetic operators for constant multi_ptr
    auto constantMultiPtrIncPost = constantMultiPtr++;
    ASSERT_RETURN_TYPE(multiPtrConstant, constantMultiPtrIncPost,
                       "cl::sycl::multi_ptr operator++(int)");
    auto constantMultiPtrIncPre = ++constantMultiPtr;
    ASSERT_RETURN_TYPE(multiPtrConstant, constantMultiPtrIncPre,
                       "cl::sycl::multi_ptr operator++()");
    auto constantMultiPtrDecPost = constantMultiPtr--;
    ASSERT_RETURN_TYPE(multiPtrConstant, constantMultiPtrDecPost,
                       "cl::sycl::multi_ptr operator--(int)");
    auto constantMultiPtrDecPre = --constantMultiPtr;
    ASSERT_RETURN_TYPE(multiPtrConstant, constantMultiPtrDecPre,
                       "cl::sycl::multi_ptr operator--()");
    auto constantMultiPtrAdd = constantMultiPtr + diff;
    ASSERT_RETURN_TYPE(multiPtrConstant, constantMultiPtrAdd,
                       "cl::sycl::multi_ptr operator+(std::ptrdiff_t r)");
    auto constantMultiPtrSub = constantMultiPtr - diff;
    ASSERT_RETURN_TYPE(multiPtrConstant, constantMultiPtrSub,
                       "cl::sycl::multi_ptr operator-(std::ptrdiff_t r)");
    auto constantMultiPtrAddAssign = constantMultiPtr += diff;
    ASSERT_RETURN_TYPE(multiPtrConstant, constantMultiPtrAddAssign,
                       "cl::sycl::multi_ptr operator+=(std::ptrdiff_t r)");
    auto constantMultiPtrSubAssign = constantMultiPtr -= diff;
    ASSERT_RETURN_TYPE(multiPtrConstant, constantMultiPtrSubAssign,
                       "cl::sycl::multi_ptr operator-=(std::ptrdiff_t r)");

    // check all arithmetic operators for local multi_ptr
    auto localMultiPtrIncPost = localMultiPtr++;
    ASSERT_RETURN_TYPE(multiPtrLocal, localMultiPtrIncPost,
                       "cl::sycl::multi_ptr operator++(int)");
    auto localMultiPtrIncPre = ++localMultiPtr;
    ASSERT_RETURN_TYPE(multiPtrLocal, localMultiPtrIncPre,
                       "cl::sycl::multi_ptr operator++()");
    auto localMultiPtrDecPost = localMultiPtr--;
    ASSERT_RETURN_TYPE(multiPtrLocal, localMultiPtrDecPost,
                       "cl::sycl::multi_ptr operator--(int)");
    auto localMultiPtrDecPre = --localMultiPtr;
    ASSERT_RETURN_TYPE(multiPtrLocal, localMultiPtrDecPre,
                       "cl::sycl::multi_ptr operator--()");
    auto localMultiPtrAdd = localMultiPtr + diff;
    ASSERT_RETURN_TYPE(multiPtrLocal, localMultiPtrAdd,
                       "cl::sycl::multi_ptr operator+(std::ptrdiff_t r)");
    auto localMultiPtrSub = localMultiPtr - diff;
    ASSERT_RETURN_TYPE(multiPtrLocal, localMultiPtrSub,
                       "cl::sycl::multi_ptr operator-(std::ptrdiff_t r)");
    auto localMultiPtrAddAssign = localMultiPtr += diff;
    ASSERT_RETURN_TYPE(multiPtrLocal, localMultiPtrAddAssign,
                       "cl::sycl::multi_ptr operator+=(std::ptrdiff_t r)");
    auto localMultiPtrSubAssign = localMultiPtr -= diff;
    ASSERT_RETURN_TYPE(multiPtrLocal, localMultiPtrSubAssign,
                       "cl::sycl::multi_ptr operator-=(std::ptrdiff_t r)");

    // check all arithmetic operators for private multi_ptr
    auto privateMultiPtrIncPost = privateMultiPtr++;
    ASSERT_RETURN_TYPE(multiPtrPrivate, privateMultiPtrIncPost,
                       "cl::sycl::multi_ptr operator++(int)");
    auto privateMultiPtrIncPre = ++privateMultiPtr;
    ASSERT_RETURN_TYPE(multiPtrPrivate, privateMultiPtrIncPre,
                       "cl::sycl::multi_ptr operator++()");
    auto privateMultiPtrDecPost = privateMultiPtr--;
    ASSERT_RETURN_TYPE(multiPtrPrivate, privateMultiPtrDecPost,
                       "cl::sycl::multi_ptr operator--(int)");
    auto privateMultiPtrDecPre = --privateMultiPtr;
    ASSERT_RETURN_TYPE(multiPtrPrivate, privateMultiPtrDecPre,
                       "cl::sycl::multi_ptr operator--()");
    auto privateMultiPtrAdd = privateMultiPtr + diff;
    ASSERT_RETURN_TYPE(multiPtrPrivate, privateMultiPtrAdd,
                       "cl::sycl::multi_ptr operator+(std::ptrdiff_t r)");
    auto privateMultiPtrSub = privateMultiPtr - diff;
    ASSERT_RETURN_TYPE(multiPtrPrivate, privateMultiPtrSub,
                       "cl::sycl::multi_ptr operator-(std::ptrdiff_t r)");
    auto privateMultiPtrAddAssign = privateMultiPtr += diff;
    ASSERT_RETURN_TYPE(multiPtrPrivate, privateMultiPtrAddAssign,
                       "cl::sycl::multi_ptr operator+=(std::ptrdiff_t r)");
    auto privateMultiPtrSubAssign = privateMultiPtr -= diff;
    ASSERT_RETURN_TYPE(multiPtrPrivate, privateMultiPtrSubAssign,
                       "cl::sycl::multi_ptr operator-=(std::ptrdiff_t r)");
  }

  template <cl::sycl::access::address_space Space>
  void relational_operators(cl::sycl::multi_ptr<U, Space> multiPtr) const {
#define TEST_RELATION_OPERATOR_TEMPLATE(OP, LHS, RHS, LHS_TY_STR, RHS_TY_STR) \
  {                                                                           \
    auto res = LHS OP RHS;                                                    \
                                                                              \
    ASSERT_RETURN_TYPE(bool, res, "cl::sycl::operator" #OP "(" LHS_TY_STR     \
                                  ", " RHS_TY_STR ")");                       \
  }

#define TEST_RELATION_OPERATOR(LHS, RHS, LHS_TY_STR, RHS_TY_STR)        \
  TEST_RELATION_OPERATOR_TEMPLATE(==, LHS, RHS, LHS_TY_STR, RHS_TY_STR) \
  TEST_RELATION_OPERATOR_TEMPLATE(!=, LHS, RHS, LHS_TY_STR, RHS_TY_STR) \
  TEST_RELATION_OPERATOR_TEMPLATE(<, LHS, RHS, LHS_TY_STR, RHS_TY_STR)  \
  TEST_RELATION_OPERATOR_TEMPLATE(>, LHS, RHS, LHS_TY_STR, RHS_TY_STR)  \
  TEST_RELATION_OPERATOR_TEMPLATE(<=, LHS, RHS, LHS_TY_STR, RHS_TY_STR) \
  TEST_RELATION_OPERATOR_TEMPLATE(>=, LHS, RHS, LHS_TY_STR, RHS_TY_STR)

    TEST_RELATION_OPERATOR(multiPtr, multiPtr, "cl::sycl::multi_ptr",
                           "cl::sycl::multi_ptr");
    TEST_RELATION_OPERATOR(nullptr, multiPtr, "std::nullptr_t",
                           "cl::sycl::multi_ptr");
    TEST_RELATION_OPERATOR(multiPtr, nullptr, "cl::sycl::multi_ptr",
                           "std::nullptr_t");
#undef TEST_RELATION_OPERATOR_TEMPLATE
#undef TEST_RELATION_OPERATOR
  }

  template <cl::sycl::access::address_space Space>
  void reference_member_types(cl::sycl::multi_ptr<U, Space> multiPtr) const {
    using multi_ptr_t = cl::sycl::multi_ptr<U, Space>;
    static_assert(
        std::is_same<multi_ptr_t, cl::sycl::multi_ptr<U, Space>>::value,
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

  template <cl::sycl::access::address_space Space>
  void member_types(cl::sycl::multi_ptr<U, Space> multiPtr) const {
    using multi_ptr_t = cl::sycl::multi_ptr<U, Space>;
    static_assert(
        std::is_same<multi_ptr_t, cl::sycl::multi_ptr<U, Space>>::value,
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

    reference_member_types(multiPtr);
  }

  template <cl::sycl::access::address_space Space>
  void address_space_member(cl::sycl::multi_ptr<U, Space> multiPtr) const {
    static constexpr cl::sycl::access::address_space addressSpace =
        cl::sycl::multi_ptr<U, Space>::address_space;
    static_assert(addressSpace == Space, "Wrong address space");
  }
};

template <typename DataT, typename VoidT>
class void_check_helper {
 public:
  using multiPtrGlobal =
      cl::sycl::multi_ptr<VoidT, cl::sycl::access::address_space::global_space>;
  using multiPtrConstant =
      cl::sycl::multi_ptr<VoidT,
                          cl::sycl::access::address_space::constant_space>;
  using multiPtrLocal =
      cl::sycl::multi_ptr<VoidT, cl::sycl::access::address_space::local_space>;
  using multiPtrPrivate =
      cl::sycl::multi_ptr<VoidT,
                          cl::sycl::access::address_space::private_space>;

  static void conversion_operators(multiPtrGlobal globalMultiPtr,
                                   multiPtrConstant constantMultiPtr,
                                   multiPtrLocal localMultiPtr,
                                   multiPtrPrivate privateMultiPtr) {
    using float_t = typename cast_keep_const<DataT, float>::type;

    using floatMultiPtrGlobal =
        cl::sycl::multi_ptr<float_t,
                            cl::sycl::access::address_space::global_space>;
    using floatMultiPtrConstant =
        cl::sycl::multi_ptr<float_t,
                            cl::sycl::access::address_space::constant_space>;
    using floatMultiPtrLocal =
        cl::sycl::multi_ptr<float_t,
                            cl::sycl::access::address_space::local_space>;
    using floatMultiPtrPrivate =
        cl::sycl::multi_ptr<float_t,
                            cl::sycl::access::address_space::private_space>;

    auto gPtr = static_cast<floatMultiPtrGlobal>(globalMultiPtr);
    auto cPtr = static_cast<floatMultiPtrConstant>(constantMultiPtr);
    auto lPtr = static_cast<floatMultiPtrLocal>(localMultiPtr);
    auto pPtr = static_cast<floatMultiPtrPrivate>(privateMultiPtr);

    ASSERT_RETURN_TYPE(
        floatMultiPtrGlobal, gPtr,
        "cl::sycl::multi_ptr operator cl::sycl::multi_ptr<float, "
        "cl::sycl::access::address_space::global_space>()");
    ASSERT_RETURN_TYPE(
        floatMultiPtrConstant, cPtr,
        "cl::sycl::multi_ptr operator cl::sycl::multi_ptr<float, "
        "cl::sycl::access::address_space::constant_space>()");
    ASSERT_RETURN_TYPE(
        floatMultiPtrLocal, lPtr,
        "cl::sycl::multi_ptr operator cl::sycl::multi_ptr<float, "
        "cl::sycl::access::address_space::local_space>()");
    ASSERT_RETURN_TYPE(
        floatMultiPtrPrivate, pPtr,
        "cl::sycl::multi_ptr operator cl::sycl::multi_ptr<float, "
        "cl::sycl::access::address_space::private_space>()");
  }

  static void prefetch_operation(multiPtrGlobal globalMultiPtr) {
    // void type does not support prefeth operation
    return;
  }

  static void access_operators(multiPtrGlobal globalMultiPtr,
                               multiPtrConstant constantMultiPtr,
                               multiPtrLocal localMultiPtr,
                               multiPtrPrivate privateMultiPtr) {
    // void type does not support access operators
    return;
  }

  static void arithmetic_operators(multiPtrGlobal globalMultiPtr,
                                   multiPtrConstant constantMultiPtr,
                                   multiPtrLocal localMultiPtr,
                                   multiPtrPrivate privateMultiPtr) {
    // void type does not support arithmetic operators
    return;
  }

  template <cl::sycl::access::address_space Space>
  static void reference_member_types(
      cl::sycl::multi_ptr<VoidT, Space> multiPtr) {
    // multi_ptr<VoidT> does not have any reference member types
  }
};

template <typename T, typename U = T>
class pointer_apis {
 public:
  using data_t = typename std::remove_const<T>::type;

  using multiPtrGlobal =
      cl::sycl::multi_ptr<U, cl::sycl::access::address_space::global_space>;
  using multiPtrConstant =
      cl::sycl::multi_ptr<U, cl::sycl::access::address_space::constant_space>;
  using multiPtrLocal =
      cl::sycl::multi_ptr<U, cl::sycl::access::address_space::local_space>;
  using multiPtrPrivate =
      cl::sycl::multi_ptr<U, cl::sycl::access::address_space::private_space>;

  void operator()(util::logger &log, cl::sycl::queue &queue) {
    const int size = 64;
    cl::sycl::range<1> range(size);
    cl::sycl::range<1> logRange(1);
    cl::sycl::unique_ptr_class<data_t[]> data(new data_t[size]);
    cl::sycl::buffer<T, 1> buffer(data.get(), range);

    queue.submit([&](cl::sycl::handler &handler) {
      cl::sycl::accessor<T, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::global_buffer>
          globalAccessor(buffer, handler);
      cl::sycl::accessor<T, 1, cl::sycl::access::mode::read,
                         cl::sycl::access::target::constant_buffer>
          constantAccessor(buffer, handler);
      cl::sycl::accessor<T, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::local>
          localAccessor(size, handler);

      handler.single_task< class kernel0<T, U>>(
            [globalAccessor, constantAccessor, localAccessor]() {
        data_t privateData[1];
        check_helper<T, U> checker;

        /** check member types
        */
        {
          // construct a set of multi_ptr
          cl::sycl::global_ptr<U> globalPtr(
              static_cast<U *>(&globalAccessor[0]));
          cl::sycl::constant_ptr<U> constantPtr(constantAccessor.get_pointer());
          cl::sycl::local_ptr<U> localPtr(static_cast<U *>(&localAccessor[0]));
          cl::sycl::private_ptr<U> privatePtr(static_cast<U *>(privateData));

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
          cl::sycl::global_ptr<U> globalPtr(
              static_cast<U *>(&globalAccessor[0]));
          cl::sycl::constant_ptr<U> constantPtr(constantAccessor.get_pointer());
          cl::sycl::local_ptr<U> localPtr(static_cast<U *>(&localAccessor[0]));
          cl::sycl::private_ptr<U> privatePtr(static_cast<U *>(privateData));

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
          cl::sycl::global_ptr<U> globalPtrA(
              static_cast<U *>(&globalAccessor[0]));
          cl::sycl::constant_ptr<U> constantPtrA(
              constantAccessor.get_pointer());
          cl::sycl::local_ptr<U> localPtrA(static_cast<U *>(&localAccessor[0]));
          cl::sycl::private_ptr<U> privatePtrA(static_cast<U *>(privateData));

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
        }

        /** check move assignment operators
        */
        {
          // construct two sets of multi_ptr
          cl::sycl::global_ptr<U> globalPtrA(
              static_cast<U *>(&globalAccessor[0]));
          cl::sycl::constant_ptr<U> constantPtrA(
              constantAccessor.get_pointer());
          cl::sycl::local_ptr<U> localPtrA(static_cast<U *>(&localAccessor[0]));
          cl::sycl::private_ptr<U> privatePtrA(static_cast<U *>(privateData));

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
        }

        /** check assigning to multi_ptr
        */
        {
          // construct a set of multi_ptr
          cl::sycl::global_ptr<U> globalPtr(
              static_cast<U *>(&globalAccessor[0]));
          cl::sycl::constant_ptr<U> constantPtr(constantAccessor.get_pointer());
          cl::sycl::local_ptr<U> localPtr(static_cast<U *>(&localAccessor[0]));
          cl::sycl::private_ptr<U> privatePtr(static_cast<U *>(privateData));

          multiPtrGlobal globalMultiPtr(globalPtr);
          multiPtrConstant constantMultiPtr(constantPtr);
          multiPtrLocal localMultiPtr(localPtr);
          multiPtrPrivate privateMultiPtr(privatePtr);

          checker.pointer_assignment(globalMultiPtr, &globalAccessor[0]);
          checker.pointer_assignment(constantMultiPtr,
                                     constantAccessor.get_pointer());
          checker.pointer_assignment(localMultiPtr, &localAccessor[0]);
          checker.pointer_assignment(privateMultiPtr, privateData);
        }

        /** check get() methods
         */
        {
          // construct a set of multi_ptr
          cl::sycl::global_ptr<U> globalPtr(
              static_cast<U *>(&globalAccessor[0]));
          cl::sycl::constant_ptr<U> constantPtr(constantAccessor.get_pointer());
          cl::sycl::local_ptr<U> localPtr(static_cast<U *>(&localAccessor[0]));
          cl::sycl::private_ptr<U> privatePtr(static_cast<U *>(privateData));

          multiPtrGlobal globalMultiPtr(globalPtr);
          multiPtrConstant constantMultiPtr(constantPtr);
          multiPtrLocal localMultiPtr(localPtr);
          multiPtrPrivate privateMultiPtr(privatePtr);

          auto gPtr = globalMultiPtr.get();
          auto cPtr = constantMultiPtr.get();
          auto lPtr = localMultiPtr.get();
          auto pPtr = privateMultiPtr.get();

          ASSERT_RETURN_TYPE(typename cl::sycl::global_ptr<U>::pointer_t, gPtr,
                             "cl::sycl::multi_ptr::get()");
          ASSERT_RETURN_TYPE(typename cl::sycl::constant_ptr<U>::pointer_t,
                             cPtr, "cl::sycl::multi_ptr::get()");
          ASSERT_RETURN_TYPE(typename cl::sycl::local_ptr<U>::pointer_t, lPtr,
                             "cl::sycl::multi_ptr::get()");
          ASSERT_RETURN_TYPE(typename cl::sycl::private_ptr<U>::pointer_t, pPtr,
                             "cl::sycl::multi_ptr::get()");
        }

        /** check prefetch() method
        */
        {
          // construct a global multi_ptr
          cl::sycl::global_ptr<U> globalPtr(
              static_cast<U *>(&globalAccessor[0]));
          multiPtrGlobal globalMultiPtr(globalPtr);
          checker.prefetch_operation(globalMultiPtr);
        }

        /** check implicit conversion to a raw pointer
         */
        {
          // construct a set of multi_ptr
          cl::sycl::global_ptr<U> globalPtr(
              static_cast<U *>(&globalAccessor[0]));
          cl::sycl::constant_ptr<U> constantPtr(constantAccessor.get_pointer());
          cl::sycl::local_ptr<U> localPtr(static_cast<U *>(&localAccessor[0]));
          cl::sycl::private_ptr<U> privatePtr(static_cast<U *>(privateData));

          multiPtrGlobal globalMultiPtr(globalPtr);
          multiPtrConstant constantMultiPtr(constantPtr);
          multiPtrLocal localMultiPtr(localPtr);
          multiPtrPrivate privateMultiPtr(privatePtr);

          U *gPtr = globalMultiPtr;
          U *cPtr = constantMultiPtr;
          U *lPtr = localMultiPtr;
          U *pPtr = privateMultiPtr;
        }

        /** check multi_ptr conversion methods
         */
        {
          // construct a set of multi_ptr
          cl::sycl::global_ptr<U> globalPtr(
              static_cast<U *>(&globalAccessor[0]));
          cl::sycl::constant_ptr<U> constantPtr(constantAccessor.get_pointer());
          cl::sycl::local_ptr<U> localPtr(static_cast<U *>(&localAccessor[0]));
          cl::sycl::private_ptr<U> privatePtr(static_cast<U *>(privateData));

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
          cl::sycl::global_ptr<U> globalPtr(
              static_cast<U *>(&globalAccessor[0]));
          cl::sycl::constant_ptr<U> constantPtr(constantAccessor.get_pointer());
          cl::sycl::local_ptr<U> localPtr(static_cast<U *>(&localAccessor[0]));
          cl::sycl::private_ptr<U> privatePtr(static_cast<U *>(privateData));

          multiPtrGlobal globalMultiPtr(globalPtr);
          multiPtrConstant constantMultiPtr(constantPtr);
          multiPtrLocal localMultiPtr(localPtr);
          multiPtrPrivate privateMultiPtr(privatePtr);

          checker.access_operators(globalMultiPtr, constantMultiPtr,
                                   localMultiPtr, privateMultiPtr);
        }

        /** check operator->() methods
        */
        {
          // construct a set of multi_ptr
          cl::sycl::global_ptr<U> globalPtr(
              static_cast<U *>(&globalAccessor[0]));
          cl::sycl::constant_ptr<U> constantPtr(constantAccessor.get_pointer());
          cl::sycl::local_ptr<U> localPtr(static_cast<U *>(&localAccessor[0]));
          cl::sycl::private_ptr<U> privatePtr(static_cast<U *>(privateData));

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
          cl::sycl::global_ptr<U> globalPtr(
              static_cast<U *>(&globalAccessor[0]));
          cl::sycl::constant_ptr<U> constantPtr(constantAccessor.get_pointer());
          cl::sycl::local_ptr<U> localPtr(static_cast<U *>(&localAccessor[0]));
          cl::sycl::private_ptr<U> privatePtr(static_cast<U *>(privateData));

          multiPtrGlobal globalMultiPtr(globalPtr);
          multiPtrConstant constantMultiPtr(constantPtr);
          multiPtrLocal localMultiPtr(localPtr);
          multiPtrPrivate privateMultiPtr(privatePtr);

          checker.arithmetic_operators(globalMultiPtr, constantMultiPtr,
                                       localMultiPtr, privateMultiPtr);
        }

        /** check make_ptr function
        */
        {
          multiPtrGlobal globalMultiPtr =
              cl::sycl::make_ptr<U,
                                 cl::sycl::access::address_space::global_space>(
                  &globalAccessor[0]);
          multiPtrConstant constantMultiPtr = cl::sycl::make_ptr<
              U, cl::sycl::access::address_space::constant_space>(
              constantAccessor.get_pointer().get());
          multiPtrLocal localMultiPtr =
              cl::sycl::make_ptr<U,
                                 cl::sycl::access::address_space::local_space>(
                  &localAccessor[0]);
          multiPtrPrivate privateMultiPtr = cl::sycl::make_ptr<
              U, cl::sycl::access::address_space::private_space>(privateData);
        }

        /** check relation functions
        */
        {
          // construct a set of multi_ptr
          cl::sycl::global_ptr<U> globalPtr(
              static_cast<U *>(&globalAccessor[0]));
          cl::sycl::constant_ptr<U> constantPtr(constantAccessor.get_pointer());
          cl::sycl::local_ptr<U> localPtr(static_cast<U *>(&localAccessor[0]));
          cl::sycl::private_ptr<U> privatePtr(static_cast<U *>(privateData));

          checker.relational_operators(globalPtr);
          checker.relational_operators(constantPtr);
          checker.relational_operators(localPtr);
          checker.relational_operators(privatePtr);
        }

      });
    });

    /** check space field assignment operators
    */
    {
      // construct two sets of multi_ptr
      cl::sycl::global_ptr<U> globalPtrA;
      cl::sycl::constant_ptr<U> constantPtrA;
      cl::sycl::local_ptr<U> localPtrA;
      cl::sycl::private_ptr<U> privatePtrA;

      static constexpr auto resG = decltype(globalPtrA)::address_space;
      static constexpr auto resC = decltype(constantPtrA)::address_space;
      static constexpr auto resL = decltype(localPtrA)::address_space;
      static constexpr auto resP = decltype(privatePtrA)::address_space;

      check_return_type<cl::sycl::access::address_space>(
          log, resG, "cl::sycl::multi_ptr::space");
      check_return_type<cl::sycl::access::address_space>(
          log, resC, "cl::sycl::multi_ptr::space");
      check_return_type<cl::sycl::access::address_space>(
          log, resL, "cl::sycl::multi_ptr::space");
      check_return_type<cl::sycl::access::address_space>(
          log, resP, "cl::sycl::multi_ptr::space");

      check_return_value<cl::sycl::access::address_space>(
          log, resG, cl::sycl::access::address_space::global_space,
          "cl::sycl::multi_ptr::space");
      check_return_value<cl::sycl::access::address_space>(
          log, resC, cl::sycl::access::address_space::constant_space,
          "cl::sycl::multi_ptr::space");
      check_return_value<cl::sycl::access::address_space>(
          log, resL, cl::sycl::access::address_space::local_space,
          "cl::sycl::multi_ptr::space");
      check_return_value<cl::sycl::access::address_space>(
          log, resP, cl::sycl::access::address_space::private_space,
          "cl::sycl::multi_ptr::space");
    }
  }
};

template <>
void check_helper<int, void>::conversion_operators(
    check_helper<int, void>::multiPtrGlobal globalMultiPtr,
    check_helper<int, void>::multiPtrConstant constantMultiPtr,
    check_helper<int, void>::multiPtrLocal localMultiPtr,
    check_helper<int, void>::multiPtrPrivate privateMultiPtr) const {
  void_check_helper<int, void>::conversion_operators(
      globalMultiPtr, constantMultiPtr, localMultiPtr, privateMultiPtr);
}
template <>
void check_helper<const int, const void>::conversion_operators(
    check_helper<const int, const void>::multiPtrGlobal globalMultiPtr,
    check_helper<const int, const void>::multiPtrConstant constantMultiPtr,
    check_helper<const int, const void>::multiPtrLocal localMultiPtr,
    check_helper<const int, const void>::multiPtrPrivate privateMultiPtr)
    const {
  void_check_helper<const int, const void>::conversion_operators(
      globalMultiPtr, constantMultiPtr, localMultiPtr, privateMultiPtr);
}

template <>
void check_helper<int, void>::prefetch_operation(
    check_helper<int, void>::multiPtrGlobal globalMultiPtr) const {
  void_check_helper<int, void>::prefetch_operation(globalMultiPtr);
}
template <>
void check_helper<const int, const void>::prefetch_operation(
    check_helper<const int, const void>::multiPtrGlobal globalMultiPtr) const {
  void_check_helper<const int, const void>::prefetch_operation(globalMultiPtr);
}

template <>
void check_helper<int, void>::access_operators(
    check_helper<int, void>::multiPtrGlobal globalMultiPtr,
    check_helper<int, void>::multiPtrConstant constantMultiPtr,
    check_helper<int, void>::multiPtrLocal localMultiPtr,
    check_helper<int, void>::multiPtrPrivate privateMultiPtr) const {
  void_check_helper<int, void>::access_operators(
      globalMultiPtr, constantMultiPtr, localMultiPtr, privateMultiPtr);
}
template <>
void check_helper<const int, const void>::access_operators(
    check_helper<const int, const void>::multiPtrGlobal globalMultiPtr,
    check_helper<const int, const void>::multiPtrConstant constantMultiPtr,
    check_helper<const int, const void>::multiPtrLocal localMultiPtr,
    check_helper<const int, const void>::multiPtrPrivate privateMultiPtr)
    const {
  void_check_helper<const int, const void>::access_operators(
      globalMultiPtr, constantMultiPtr, localMultiPtr, privateMultiPtr);
}

template <>
void check_helper<int, void>::arithmetic_operators(
    check_helper<int, void>::multiPtrGlobal globalMultiPtr,
    check_helper<int, void>::multiPtrConstant constantMultiPtr,
    check_helper<int, void>::multiPtrLocal localMultiPtr,
    check_helper<int, void>::multiPtrPrivate privateMultiPtr) const {
  void_check_helper<int, void>::arithmetic_operators(
      globalMultiPtr, constantMultiPtr, localMultiPtr, privateMultiPtr);
}
template <>
void check_helper<const int, const void>::arithmetic_operators(
    check_helper<const int, const void>::multiPtrGlobal globalMultiPtr,
    check_helper<const int, const void>::multiPtrConstant constantMultiPtr,
    check_helper<const int, const void>::multiPtrLocal localMultiPtr,
    check_helper<const int, const void>::multiPtrPrivate privateMultiPtr)
    const {
  void_check_helper<const int, const void>::arithmetic_operators(
      globalMultiPtr, constantMultiPtr, localMultiPtr, privateMultiPtr);
}

template <>
template <cl::sycl::access::address_space Space>
void check_helper<int, void>::reference_member_types(
    cl::sycl::multi_ptr<void, Space> multiPtr) const {
  void_check_helper<int, void>::reference_member_types(multiPtr);
}
template <>
template <cl::sycl::access::address_space Space>
void check_helper<const int, const void>::reference_member_types(
    cl::sycl::multi_ptr<const void, Space> multiPtr) const {
  void_check_helper<const int, const void>::reference_member_types(multiPtr);
}

template <>
void check_helper<user_struct>::arrow_operators(
    check_helper<user_struct>::multiPtrGlobal globalMultiPtr,
    check_helper<user_struct>::multiPtrConstant constantMultiPtr,
    check_helper<user_struct>::multiPtrLocal localMultiPtr,
    check_helper<user_struct>::multiPtrPrivate privateMultiPtr) const {
  auto globalElem = globalMultiPtr->a;
  auto constantElem = constantMultiPtr->a;
  auto localElem = localMultiPtr->a;
  auto privateElem = privateMultiPtr->a;

  ASSERT_RETURN_TYPE(float, globalElem, "cl::sycl::multi_ptr operator->()");
  ASSERT_RETURN_TYPE(float, constantElem, "cl::sycl::multi_ptr operator->()");
  ASSERT_RETURN_TYPE(float, localElem, "cl::sycl::multi_ptr operator->()");
  ASSERT_RETURN_TYPE(float, privateElem, "cl::sycl::multi_ptr operator->()");
}

/** tests the api for explicit pointers
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute this test
  */
  void run(util::logger &log) override {
    try {
      auto queue = util::get_cts_object::queue();

      pointer_apis<int, void> voidTests;
      voidTests(log, queue);

      pointer_apis<const int, const void> constVoidTests;
      constVoidTests(log, queue);

      pointer_apis<char> charTests;
      charTests(log, queue);

      pointer_apis<const char> constCharTests;
      constCharTests(log, queue);

      pointer_apis<short> shortTests;
      shortTests(log, queue);

      pointer_apis<const short> constShortTests;
      constShortTests(log, queue);

      pointer_apis<int> intTests;
      intTests(log, queue);

      pointer_apis<const int> constIntTests;
      constIntTests(log, queue);

      pointer_apis<long> longTests;
      longTests(log, queue);

      pointer_apis<const long> constLongTests;
      constLongTests(log, queue);

      pointer_apis<long long> longLongTests;
      longLongTests(log, queue);

      pointer_apis<const long long> constLongLongTests;
      constLongLongTests(log, queue);

      pointer_apis<float> floatTests;
      floatTests(log, queue);

      pointer_apis<const float> constFloatTests;
      constFloatTests(log, queue);

      pointer_apis<double> doubleTests;
      doubleTests(log, queue);

      pointer_apis<const double> constDoubleTests;
      constDoubleTests(log, queue);

      pointer_apis<user_struct> userStructTests;
      userStructTests(log, queue);

      pointer_apis<const user_struct> constUserStructTests;
      constUserStructTests(log, queue);

      queue.wait_and_throw();
    } catch (const cl::sycl::exception &e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAME */

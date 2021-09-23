/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Common code for multi_ptr constructors' tests
//
*******************************************************************************/

#ifndef SYCL_CTS_TEST_MULTI_PTR_MULTI_PTR_CONSTRUCTORS_COMMON_H
#define SYCL_CTS_TEST_MULTI_PTR_MULTI_PTR_CONSTRUCTORS_COMMON_H

#include "../common/common.h"
#include "multi_ptr_common.h"
using namespace multi_ptr_common;

namespace multi_ptr_constructors_common {
using namespace sycl_cts;

template <typename T, typename U>
class kernel0;

template <typename T, typename U = T>
class pointer_ctors {
 public:
  using data_t = typename std::remove_const<T>::type;

  using multiPtrGlobal =
      sycl::multi_ptr<U, sycl::access::address_space::global_space>;
  using multiPtrConstant =
      sycl::multi_ptr<U, sycl::access::address_space::constant_space>;
  using multiPtrLocal =
      sycl::multi_ptr<U, sycl::access::address_space::local_space>;
  using multiPtrPrivate =
      sycl::multi_ptr<U, sycl::access::address_space::private_space>;

  void operator()(sycl::queue &queue, const std::string& dataTypeName) {
    return operator() (queue, dataTypeName, dataTypeName);
  }
  void operator()(sycl::queue &queue, const std::string&,
                  const std::string&) {
    const int size = 64;
    sycl::range<1> range(size);
    std::unique_ptr<data_t[]> data(new data_t[size]);
    sycl::buffer<T, 1> buffer(data.get(), range);

    queue.submit([&](sycl::handler &handler) {
      sycl::accessor<T, 1, sycl::access_mode::read_write,
                         sycl::target::device>
          globalAccessor(buffer, handler);
      sycl::accessor<T, 1, sycl::access_mode::read,
                         sycl::target::constant_buffer>
          constantAccessor(buffer, handler);
      sycl::accessor<T, 1, sycl::access_mode::read_write,
                         sycl::target::local>
          localAccessor(size, handler);

      handler.single_task<class kernel0<T, U>>([=] {
        data_t privateData[1];

        /** check default constructors
         */
        {
          multiPtrGlobal globalMultiPtr;
          multiPtrConstant constantMultiPtr;
          multiPtrLocal localMultiPtr;
          multiPtrPrivate privateMultiPtr;

          silence_warnings(globalMultiPtr, constantMultiPtr, localMultiPtr,
                           privateMultiPtr);
        }

        /** check (elementType *) constructors
         */
        {
          sycl::global_ptr<U> globalPtr(
              static_cast<U *>(&globalAccessor[0]));
          sycl::constant_ptr<U> constantPtr(constantAccessor.get_pointer());
          sycl::local_ptr<U> localPtr(static_cast<U *>(&localAccessor[0]));
          sycl::private_ptr<U> privatePtr(static_cast<U *>(privateData));

          multiPtrGlobal globalMultiPtr(globalPtr);
          multiPtrConstant constantMultiPtr(constantPtr);
          multiPtrLocal localMultiPtr(localPtr);
          multiPtrPrivate privateMultiPtr(privatePtr);

          silence_warnings(globalMultiPtr, constantMultiPtr, localMultiPtr,
                           privateMultiPtr);
        }

        /** check (pointer) constructors
         */
        {
          sycl::global_ptr<U> globalPtr(
              static_cast<U *>(&globalAccessor[0]));
          sycl::constant_ptr<U> constantPtr(constantAccessor.get_pointer());
          sycl::local_ptr<U> localPtr(static_cast<U *>(&localAccessor[0]));
          sycl::private_ptr<U> privatePtr(static_cast<U *>(privateData));

          multiPtrGlobal globalMultiPtr(globalPtr.get());
          multiPtrConstant constantMultiPtr(constantPtr.get());
          multiPtrLocal localMultiPtr(localPtr.get());
          multiPtrPrivate privateMultiPtr(privatePtr.get());

          silence_warnings(globalMultiPtr, constantMultiPtr, localMultiPtr,
                           privateMultiPtr);
        }

        /** check (std::nullptr_t) constructors
         */
        {
          multiPtrGlobal globalMultiPtr(nullptr);
          multiPtrConstant constantMultiPtr(nullptr);
          multiPtrLocal localMultiPtr(nullptr);
          multiPtrPrivate privateMultiPtr(nullptr);

          silence_warnings(globalMultiPtr, constantMultiPtr, localMultiPtr,
                           privateMultiPtr);
        }

        /** check (accessor) constructors
         */
        {
          multiPtrGlobal globalMultiPtr(globalAccessor);
          multiPtrConstant constantMultiPtr(constantAccessor);
          multiPtrLocal localMultiPtr(localAccessor);

          silence_warnings(globalMultiPtr, constantMultiPtr, localMultiPtr);
        }

        /** check copy constructors
         */
        {
          sycl::global_ptr<U> globalPtrA(
              static_cast<U *>(&globalAccessor[0]));
          sycl::constant_ptr<U> constantPtrA(
              constantAccessor.get_pointer());
          sycl::local_ptr<U> localPtrA(static_cast<U *>(&localAccessor[0]));
          sycl::private_ptr<U> privatePtrA(static_cast<U *>(privateData));

          multiPtrGlobal globalMultiPtrA(globalPtrA);
          multiPtrConstant constantMultiPtrA(constantPtrA);
          multiPtrLocal localMultiPtrA(localPtrA);
          multiPtrPrivate privateMultiPtrA(privatePtrA);

          multiPtrGlobal globalMultiPtrB(globalMultiPtrA);
          multiPtrConstant constantMultiPtrB(constantMultiPtrA);
          multiPtrLocal localMultiPtrB(localMultiPtrA);
          multiPtrPrivate privateMultiPtrB(privateMultiPtrA);

          silence_warnings(globalMultiPtrB, constantMultiPtrB, localMultiPtrB,
                           privateMultiPtrB);
        }

        /** check move constructors
         */
        {
          sycl::global_ptr<U> globalPtrA(
              static_cast<U *>(&globalAccessor[0]));
          sycl::constant_ptr<U> constantPtrA(
              constantAccessor.get_pointer());
          sycl::local_ptr<U> localPtrA(static_cast<U *>(&localAccessor[0]));
          sycl::private_ptr<U> privatePtrA(static_cast<U *>(privateData));

          multiPtrGlobal globalMultiPtrA(globalPtrA);
          multiPtrConstant constantMultiPtrA(constantPtrA);
          multiPtrLocal localMultiPtrA(localPtrA);
          multiPtrPrivate privateMultiPtrA(privatePtrA);

          multiPtrGlobal globalMultiPtrB = std::move(globalMultiPtrA);
          multiPtrConstant constantMultiPtrB = std::move(constantMultiPtrA);
          multiPtrLocal localMultiPtrB = std::move(localMultiPtrA);
          multiPtrPrivate privateMultiPtrB = std::move(privateMultiPtrA);

          silence_warnings(globalMultiPtrB, constantMultiPtrB, localMultiPtrB,
                           privateMultiPtrB);
        }

      });
    });
  }
};

template <typename T>
using check_pointer_ctors = check_pointer<pointer_ctors, T>;

template <typename T>
using check_void_pointer_ctors = check_void_pointer<pointer_ctors, T>;

}  // namespace multi_ptr_constructors_common

#endif  // SYCL_CTS_TEST_MULTI_PTR_MULTI_PTR_CONSTRUCTORS_COMMON_H

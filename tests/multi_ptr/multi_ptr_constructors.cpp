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

#define TEST_NAME multi_ptr_constructors

namespace TEST_NAME {
using namespace sycl_cts;

template <typename T, typename U>
class kernel0;

struct user_struct {
  float a;
  int b;
  char c;
};

template <typename T, typename U = T>
class pointer_ctors {
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

  void operator()(cl::sycl::queue &queue) {
    const int size = 64;
    cl::sycl::range<1> range(size);
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

      handler.single_task<class kernel0<T, U>>([=]() {
        data_t privateData[1];

        /** check default constructors
        */
        {
          multiPtrGlobal globalMultiPtr;
          multiPtrConstant constantMultiPtr;
          multiPtrLocal localMultiPtr;
          multiPtrPrivate privateMultiPtr;
        }

        /** check (elementType *) constructors
        */
        {
          cl::sycl::global_ptr<U> globalPtr(
              static_cast<U *>(&globalAccessor[0]));
          cl::sycl::constant_ptr<U> constantPtr(constantAccessor.get_pointer());
          cl::sycl::local_ptr<U> localPtr(static_cast<U *>(&localAccessor[0]));
          cl::sycl::private_ptr<U> privatePtr(static_cast<U *>(privateData));

          multiPtrGlobal globalMultiPtr(globalPtr);
          multiPtrConstant constantMultiPtr(constantPtr);
          multiPtrLocal localMultiPtr(localPtr);
          multiPtrPrivate privateMultiPtr(privatePtr);
        }

        /** check (pointer) constructors
        */
        {
          cl::sycl::global_ptr<U> globalPtr(
              static_cast<U *>(&globalAccessor[0]));
          cl::sycl::constant_ptr<U> constantPtr(constantAccessor.get_pointer());
          cl::sycl::local_ptr<U> localPtr(static_cast<U *>(&localAccessor[0]));
          cl::sycl::private_ptr<U> privatePtr(static_cast<U *>(privateData));

          multiPtrGlobal globalMultiPtr(globalPtr.get());
          multiPtrConstant constantMultiPtr(constantPtr.get());
          multiPtrLocal localMultiPtr(localPtr.get());
          multiPtrPrivate privateMultiPtr(privatePtr.get());
        }

        /** check (std::nullptr_t) constructors
        */
        {
          multiPtrGlobal globalMultiPtr(nullptr);
          multiPtrConstant constantMultiPtr(nullptr);
          multiPtrLocal localMultiPtr(nullptr);
          multiPtrPrivate privateMultiPtr(nullptr);
        }

        /** check (accessor) constructors
        */
        {
          multiPtrGlobal globalMultiPtr(globalAccessor);
          multiPtrConstant constantMultiPtr(constantAccessor);
          multiPtrLocal localMultiPtr(localAccessor);
        }

        /** check copy constructors
        */
        {
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

          multiPtrGlobal globalMultiPtrB(globalMultiPtrA);
          multiPtrConstant constantMultiPtrB(constantMultiPtrA);
          multiPtrLocal localMultiPtrB(localMultiPtrA);
          multiPtrPrivate privateMultiPtrB(privateMultiPtrA);
        }

        /** check move constructors
        */
        {
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

          multiPtrGlobal globalMultiPtrB = std::move(globalMultiPtrA);
          multiPtrConstant constantMultiPtrB = std::move(constantMultiPtrA);
          multiPtrLocal localMultiPtrB = std::move(localMultiPtrA);
          multiPtrPrivate privateMultiPtrB = std::move(privateMultiPtrA);
        }

      });
    });
  }
};

/** tests the constructors for explicit pointers
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

      pointer_ctors<int, void> voidTests;
      voidTests(queue);

      pointer_ctors<const int, const void> constVoidTests;
      constVoidTests(queue);

      pointer_ctors<char> charTests;
      charTests(queue);

      pointer_ctors<const char> constCharTests;
      constCharTests(queue);

      pointer_ctors<short> shortTests;
      shortTests(queue);

      pointer_ctors<const short> constShortTests;
      constShortTests(queue);

      pointer_ctors<int> intTests;
      intTests(queue);

      pointer_ctors<const int> constIntTests;
      constIntTests(queue);

      pointer_ctors<long> longTests;
      longTests(queue);

      pointer_ctors<const long> constLongTests;
      constLongTests(queue);

      pointer_ctors<long long> longLongTests;
      longLongTests(queue);

      pointer_ctors<const long long> constLongLongTests;
      constLongLongTests(queue);

      pointer_ctors<float> floatTests;
      floatTests(queue);

      pointer_ctors<const float> constFloatTests;
      constFloatTests(queue);

      pointer_ctors<double> doubleTests;
      doubleTests(queue);

      pointer_ctors<const double> constDoubleTests;
      constDoubleTests(queue);

      pointer_ctors<user_struct> userStructTests;
      userStructTests(queue);

      pointer_ctors<const user_struct> constUserStructTests;
      constUserStructTests(queue);

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

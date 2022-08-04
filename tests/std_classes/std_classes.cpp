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

#define TEST_NAME std_classes

namespace std_classes__ {
using namespace sycl_cts;

/** check vector_class
*/
template <class T, class Alloc>
using vectorClass = cl::sycl::vector_class<T, Alloc>;

/** check string_class
*/
using stringClass = cl::sycl::string_class;

/** check function_class
*/
template <class R, class... Args>
using functionClass = cl::sycl::function_class<R(Args...)>;

/** check mutex_class
*/
using mutexClass = cl::sycl::mutex_class;

/** check unique_ptr_class
*/
template <class T, class D>
using uniquePtrClass = cl::sycl::unique_ptr_class<T, D>;

/** check shared_ptr_class
*/
template <class T>
using sharedPtrClass = cl::sycl::shared_ptr_class<T>;

/** check weak_ptr_class
*/
template <class T>
using weakPtrClass = cl::sycl::weak_ptr_class<T>;

/** check hash_class
*/
template <class T>
using hashClass = cl::sycl::hash_class<T>;

/** check exception_ptr_class
*/
using exceptionPtrClass = cl::sycl::exception_ptr_class;

struct custom_deleter {
  void operator()(int *p) const {};
};

/** tests the availability of std classes
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
    /* Try instantiating these classes */
    {
      /** check vector_class
      */
      cl::sycl::vector_class<int> vector;

      /** check string_class
      */
      stringClass string;

      /** check function_class
      */
      functionClass<void> function;

      /** check mutex_class
      */
      mutexClass mutex;

      /** check unique_ptr_class
      */
      uniquePtrClass<int, custom_deleter> uniquePtr;

      /** check shared_ptr_class
      */
      sharedPtrClass<int> sharedPtr;

      /** check weak_ptr_class
      */
      weakPtrClass<int> weakPtr;

      /** check hash_class
      */
      hashClass<int> hash;

      /** check exception_ptr_class
      */
      exceptionPtrClass exceptionPtr;
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace std_classes__ */

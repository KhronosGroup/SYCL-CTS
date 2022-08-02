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

#ifndef __SYCLCTS_UTIL_PROXY_H
#define __SYCLCTS_UTIL_PROXY_H

#include "test_base.h"

namespace sycl_cts {
namespace util {

// test harness function to register a given test
// defined in collection.cpp
extern void register_test(test_base *test);

/** test proxy class
 *  this class is used to register tests with the test harness at compile time.
 */
template <typename T>
class test_proxy {
 public:
  /** test_proxy constructor
   */
  test_proxy() {
    // use an externed function to cut dependency on the collection
    register_test(new T());
  }
};

}  // namespace util
}  // namespace sycl_cts

#endif  // __SYCLCTS_UTIL_PROXY_H
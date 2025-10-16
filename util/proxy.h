/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2021-2022 The Khronos Group Inc.
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

#include <string>

#include <catch2/interfaces/catch_interfaces_registry_hub.hpp>
#include <catch2/internal/catch_test_registry.hpp>

#include "test_base.h"

namespace sycl_cts {
namespace util {

/**
 * Utility class for registering legacy test cases.
 * @deprecated Use Catch2's TEST_CASE macro instead.
 */
template <typename T>
class test_proxy {
 public:
  test_proxy() {
    T{}.get_info_legacy(m_info);

    Catch::AutoReg(
        Catch::makeTestInvoker<T>(&T::run_legacy),
        {m_info.m_file.c_str(), /*.line=*/0},
        "__SYCL_CTS_LEGACY_TEST__" + std::to_string(next_legacy_test_id++),
        {m_info.m_name, "[legacy]"});
  }

 private:
  test_base::info m_info;
  inline static size_t next_legacy_test_id = 0;
};

}  // namespace util
}  // namespace sycl_cts

#endif  // __SYCLCTS_UTIL_PROXY_H

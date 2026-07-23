/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2017-2022 Codeplay Software LTD.
//  SPDX-FileCopyrightText: 2021-2022 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
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

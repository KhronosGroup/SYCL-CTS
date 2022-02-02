/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2021 The Khronos Group Inc.
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
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
    test_base::info info;
    T{}.get_info_legacy(info);

    Catch::AutoReg(
        Catch::makeTestInvoker<T>(&T::run_legacy), {__FILE__, __LINE__},
        "__SYCL_CTS_LEGACY_TEST__" + std::to_string(next_legacy_test_id++),
        {info.m_name, "[legacy]"});
  }

 private:
  inline static size_t next_legacy_test_id = 0;
};

}  // namespace util
}  // namespace sycl_cts

#endif  // __SYCLCTS_UTIL_PROXY_H
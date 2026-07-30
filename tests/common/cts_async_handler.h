/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2017-2022 Codeplay Software LTD.
//  SPDX-FileCopyrightText: 2022 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_COMMON_CTS_ASYNC_HANDLER_H
#define __SYCLCTS_TESTS_COMMON_CTS_ASYNC_HANDLER_H

#include <sycl/sycl.hpp>

// Change of async handler can affect on tests of optional kernel features
struct cts_async_handler {
  void operator()(sycl::exception_list l) {
    for (auto &e : l) {
      std::rethrow_exception(e);
    }
  }
};

#endif  // __SYCLCTS_TESTS_COMMON_CTS_ASYNC_HANDLER_H

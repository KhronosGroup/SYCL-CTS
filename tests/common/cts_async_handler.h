/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_COMMON_CTS_ASYNC_HANDLER_H
#define __SYCLCTS_TESTS_COMMON_CTS_ASYNC_HANDLER_H

#include <sycl/sycl.hpp>

struct cts_async_handler {
  void operator()(sycl::exception_list l) {
    for (auto &e : l) {
      std::rethrow_exception(e);
    }
  }
};

#endif  // __SYCLCTS_TESTS_COMMON_CTS_ASYNC_HANDLER_H

/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#pragma once

#include "sycl.h"

struct cts_async_handler {
  void operator()(cl::sycl::exception_list l) {
    for (auto &e : l) {
      std::rethrow_exception(e);
    }
  }
};

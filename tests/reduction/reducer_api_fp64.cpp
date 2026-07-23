/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/
#include "../common/common.h"
#include "../common/disabled_for_test_case.h"
// FIXME: re-enable when sycl::reduction is implemented in AdaptiveCpp
#if !SYCL_CTS_COMPILING_WITH_ADAPTIVECPP
#include "reducer_api.h"
#endif

#include <string>
// FIXME: re-enable when reducer is fully implemented in AdaptiveCpp
DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("reducer api fp64", "[reducer][fp64]")({
  sycl::queue queue = sycl_cts::util::get_cts_object::queue();
  if (!queue.get_device().has(sycl::aspect::fp64)) {
    SKIP("Device does not support double precision floating point operations.");
  }

  using type = double;
  const std::string type_name("double");
  check_reducer_subscript<type>{}(queue, type_name);
  check_reducer_identity<type>{}(queue, type_name);
});

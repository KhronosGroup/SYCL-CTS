/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/semantics_reference.h"

struct storage {
  std::size_t device_count;

  explicit storage(const sycl::context& context)
      : device_count(context.get_devices().size()) {}

  bool check(const sycl::context& context) const {
    return context.get_devices().size() == device_count;
  }
};

TEST_CASE("context common reference semantics", "[context]") {
  sycl::context context_0{};
  sycl::context context_1{};

  common_reference_semantics::check_host<storage>(context_0, context_1,
                                                  "context");
}

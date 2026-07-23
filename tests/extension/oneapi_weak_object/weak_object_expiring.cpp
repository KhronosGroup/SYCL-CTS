/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "weak_object_common.h"

namespace weak_object_expiring {
using namespace weak_object_common;

template <typename SYCLObjT>
void test_weak_object_expiring(const std::string& typeName) {
  INFO(typeName + " type: ");
  sycl::ext::oneapi::weak_object<SYCLObjT> w;
  {
    auto sycl_object = get_sycl_object<SYCLObjT>();
    w = sycl_object;
    INFO("Check that weak_object now has a value and not expired");
    CHECK(!w.expired());
  }
  INFO("Check that weak_object was expired");
  CHECK(w.expired());
}

TEST_CASE("weak_object expiring", "[weak_object]") {
#if !defined SYCL_EXT_ONEAPI_WEAK_OBJECT
  SKIP("SYCL_EXT_ONEAPI_WEAK_OBJECT is not defined");
#else
  test_weak_object_expiring<sycl::buffer<int>>("buffer");
  test_weak_object_expiring<sycl::accessor<int>>("accessor");
  test_weak_object_expiring<sycl::host_accessor<int>>("host_accessor");
  test_weak_object_expiring<sycl::queue>("queue");
  test_weak_object_expiring<sycl::context>("context");
  test_weak_object_expiring<sycl::event>("event");
#endif
}
}  // namespace weak_object_expiring

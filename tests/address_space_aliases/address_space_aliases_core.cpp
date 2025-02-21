/*******************************************************************************
//
//  SYCL Next Conformance Test Suite
//
//  Copyright (c) 2025 The Khronos Group Inc.
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

#define TEST_NAME address_space_aliases_core

#include "../common/common.h"

namespace TEST_NAMESPACE {
using namespace sycl_cts;

class TEST_NAME : public sycl_cts::util::test_base {
 public:
  void get_info(test_base::info& out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  void run(util::logger& log) override {
    CHECK(std::is_same_v<sycl::access::address_space, sycl::addrspace>);

    CHECK(
        std::is_same_v<sycl::addrspace::global_space, sycl::addrspace_global>);

    CHECK(std::is_same_v<sycl::addrspace::local_space, sycl::addrspace_local>);

    CHECK(std::is_same_v<sycl::addrspace::private_space,
                         sycl::addrspace_private>);

    CHECK(std::is_same_v<sycl::addrspace::generic_space,
                         sycl::addrspace_generic>);
  }
};

util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE

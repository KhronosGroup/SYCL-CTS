/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2018-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2022-2023 The Khronos Group Inc.
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

#include "../common/common.h"
#include "../common/invoke.h"
#include "../common/semantics_by_value.h"

#define TEST_NAME nd_item_equality

namespace TEST_NAMESPACE {
using namespace sycl_cts;

template <int numDims>
struct nd_item_setup_kernel;

template <int numDims>
struct nd_item_equality_kernel;

/** test sycl::device initialization
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info& out) const final {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  template <int numDims>
  void test_equality(util::logger& log) {
    {
      using item_t = sycl::nd_item<numDims>;

      // nd_item is not default constructible, store two objects into the array
      static constexpr size_t numItems = 2;
      using setup_kernel_t = nd_item_setup_kernel<numDims>;
      auto items =
          store_instances<numItems, invoke_nd_item<numDims, setup_kernel_t>>();

      // Check nd_item equality operator on the device side
      common_by_value_semantics::on_device_checker<item_t>::template run<
          nd_item_equality_kernel<numDims>>(
          log, items, "nd_item " + std::to_string(numDims) + " (device)");

      // Check nd_item equality operator on the host side
      common_by_value_semantics::check_on_host(
          log, items[0], "nd_item " + std::to_string(numDims) + " (host)");
    }
  }

  /** execute the test
   */
  void run(util::logger& log) final {
    test_equality<1>(log);
    test_equality<2>(log);
    test_equality<3>(log);
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE

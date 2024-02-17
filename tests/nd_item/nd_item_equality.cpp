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
    using item_t = sycl::nd_item<numDims>;
    using kernel_t = nd_item_equality_kernel<numDims>;

    // Store comparison results from kernel into a success array
    std::array<bool,
               to_integral(common_by_value_semantics::current_check::size)>
        success;
    std::fill(std::begin(success), std::end(success), true);

    {
      sycl::buffer<bool> successBuf(success.data(),
                                    sycl::range<1>(success.size()));

      const auto oneElemRange =
          util::get_cts_object::range<numDims>::get(1, 1, 1);

      auto queue = util::get_cts_object::queue();
      queue
          .submit([&](sycl::handler& cgh) {
            auto successAcc =
                successBuf.get_access<sycl::access_mode::write>(cgh);

            cgh.parallel_for<kernel_t>(
                sycl::nd_range<numDims>(oneElemRange, oneElemRange),
                [=](item_t item) {
                  common_by_value_semantics::check_equality(item, successAcc);
                });
          })
          .wait_and_throw();
    }

    for (int i = 0; i < success.size(); ++i) {
      INFO(std::string(TOSTRING(TEST_NAME)) + " is " +
           common_by_value_semantics::get_error_string(i));
      CHECK(success[i]);
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

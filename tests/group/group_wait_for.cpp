/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2018-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2022 The Khronos Group Inc.
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
#include "../common/async_work_group_copy.h"
#include "../common/invoke.h"

#define TEST_NAME group_wait_for

namespace TEST_NAMESPACE {
using namespace sycl_cts;

template <int dim>
class wait_for_kernel;

template<int dim>
struct check_dim {
  void operator()(sycl::queue &queue, sycl_cts::util::logger &log) {
    using dataT = int;
    using kernelT = wait_for_kernel<dim>;
    using kernelInvokeT = invoke_group<dim, kernelT>;
    static const std::string instanceName = "group";

    test_wait_for<kernelInvokeT, dataT>(queue, log, instanceName);
  }
};

/** test sycl::group::wait_for
*/
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
  *  @param info, test_base::info structure as output
  */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
  *  @param log, test transcript logging class
  */
  void run(util::logger &log) override {
    {
      auto queue = util::get_cts_object::queue();

      check_all_dims<check_dim>(queue, log);
    }
  }
};

util::test_proxy<TEST_NAME> proxy;

} /* namespace group_wait_for__ */

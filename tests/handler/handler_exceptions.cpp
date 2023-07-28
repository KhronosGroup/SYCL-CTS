/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
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

#include "../../util/sycl_exceptions.h"

#include "../common/common.h"

#include "catch2/catch_test_macros.hpp"

namespace handler_exceptions {

using AccT =
    sycl::accessor<int, 1, sycl::access_mode::write, sycl::target::device>;

TEST_CASE("handler require() exception", "[handler]") {
  auto queue = sycl_cts::util::get_cts_object::queue();
  auto action = [&] {
    AccT empty_acc;
    queue.submit([&](sycl::handler& cgh) { cgh.require(empty_acc); })
        .wait_and_throw();
  };

  CHECK_THROWS_MATCHES(action(), sycl::exception,
                       sycl_cts::util::equals_exception(sycl::errc::invalid));
}
}  // namespace handler_exceptions

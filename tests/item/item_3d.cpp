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
//  Provides api tests for sycl::item<1> tests
//
*******************************************************************************/

#include "../common/common.h"
#include <item_common.h>

namespace item_3d_test {

TEST_CASE("sycl::item<3> api", "[item]") {
  sycl::range<3> dataRange(64, 64, 64);
  item_common_test::test_item<3>(dataRange);
}

}  // namespace item_3d_test

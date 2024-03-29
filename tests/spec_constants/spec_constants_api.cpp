/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2023 The Khronos Group Inc.
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

TEST_CASE("specialization_id api", "[specialization_constants]") {
  using spec_id_type = sycl::specialization_id<int>;
  STATIC_CHECK_FALSE(std::is_copy_constructible_v<spec_id_type>);
  STATIC_CHECK_FALSE(std::is_move_constructible_v<spec_id_type>);
  STATIC_CHECK_FALSE(std::is_copy_assignable_v<spec_id_type>);
  STATIC_CHECK_FALSE(std::is_move_assignable_v<spec_id_type>);
}

/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
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

#include "../../common/common.h"

namespace max_num_work_groups_macro::tests {
    TEST_CASE(
        "the max_num_work_groups extension defines the "
        "SYCL_KHR_MAX_NUM_WORK_GROUPS macro",
        "[khr_max_num_work_groups]") {
    #ifndef SYCL_KHR_MAX_NUM_WORK_GROUPS
    static_assert(false, "SYCL_KHR_MAX_NUM_WORK_GROUPS is not defined");
    #endif
    }
}  // namespace max_num_work_groups_macro::tests

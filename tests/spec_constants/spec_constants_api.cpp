/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
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

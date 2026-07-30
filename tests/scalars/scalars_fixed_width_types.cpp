/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2022-2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/type_coverage.h"
#include <cstdint>

static constexpr size_t types_count = 8;
class fixed_width_types_kernel;

TEST_CASE("Fixed width types size equality", "[scalars]") {
  auto queue = sycl_cts::util::get_cts_object::queue();

  std::array<const char*, types_count> types_str = {
      "int8_t",  "int16_t",  "int32_t",  "int64_t",
      "uint8_t", "uint16_t", "uint32_t", "uint64_t"};

  std::array<bool, types_count> results;

  results[0] = sizeof(int8_t) == 1;
  results[1] = sizeof(int16_t) == 2;
  results[2] = sizeof(int32_t) == 4;
  results[3] = sizeof(int64_t) == 8;
  results[4] = sizeof(uint8_t) == 1;
  results[5] = sizeof(uint16_t) == 2;
  results[6] = sizeof(uint32_t) == 4;
  results[7] = sizeof(uint64_t) == 8;

  // verify host results
  for (int i = 0; i < types_count; i++) {
    INFO("Check " << types_str[i] << " size on host");
    CHECK(results[i]);
  }

  std::fill(results.begin(), results.end(), false);

  {
    sycl::buffer<bool, 1> res_buf(results.data(), {types_count});
    queue.submit([&](sycl::handler& cgh) {
      auto res_acc =
          res_buf.template get_access<sycl::access_mode::read_write>(cgh);

      cgh.single_task<fixed_width_types_kernel>([=] {
        res_acc[0] = sizeof(int8_t) == 1;
        res_acc[1] = sizeof(int16_t) == 2;
        res_acc[2] = sizeof(int32_t) == 4;
        res_acc[3] = sizeof(int64_t) == 8;
        res_acc[4] = sizeof(uint8_t) == 1;
        res_acc[5] = sizeof(uint16_t) == 2;
        res_acc[6] = sizeof(uint32_t) == 4;
        res_acc[7] = sizeof(uint64_t) == 8;
      });
    });
    queue.wait_and_throw();
  }

  // verify device results
  for (int i = 0; i < types_count; i++) {
    INFO("Check " << types_str[i] << " size on device");
    CHECK(results[i]);
  }
}

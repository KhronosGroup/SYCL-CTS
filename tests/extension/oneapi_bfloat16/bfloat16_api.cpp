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

#include "../../common/common.h"

using bfloat16 = sycl::ext::oneapi::bfloat16;

TEST_CASE("Common interface members", "[bfloat16]") {
  SECTION("default constructor/destructor and copy") {
    CHECK(std::is_default_constructible_v<bfloat16>);
    CHECK(std::is_copy_constructible_v<bfloat16>);
    CHECK(std::is_destructible_v<bfloat16>);
    CHECK(std::is_copy_assignable_v<bfloat16>);
  }

  SECTION("float/sycl::half constructors and assignment operators") {
    float f = 1.f;
    bfloat16 bf1_float(f);
    bfloat16 bf2_float = 0;
    bf2_float = f;
    {
      INFO("Check float");
      CHECK(bf1_float == f);
      CHECK(bf2_float == f);
    }

    sycl::half hf = 1.0;
    bfloat16 bf1_half(hf);
    bfloat16 bf2_half = 0;
    bf2_half = hf;
    {
      INFO("Check sycl::half");
      CHECK(bf1_half == hf);
      CHECK(bf2_half == hf);
    }
  }

  SECTION("Conversion") {
    CHECK(std::is_convertible_v<bfloat16, float>);
    CHECK(std::is_convertible_v<bfloat16, sycl::half>);
    CHECK(std::is_convertible_v<bfloat16, bool>);
    CHECK(std::is_convertible_v<float, bfloat16>);
    CHECK(std::is_convertible_v<sycl::half, bfloat16>);
  }
}

TEST_CASE("Special values", "[bfloat16]") {
  SECTION("Check that bfloat16 occupies 16 bits of memory") {
    REQUIRE(sizeof(bfloat16) == 2);
  }

  SECTION("Check minimum positive normal value") {
    bfloat16 bf_min = sycl::bit_cast<bfloat16>(uint16_t(0b0000000010000000));

    CHECK(bf_min == std::numeric_limits<float>::min());
  }

  SECTION("NaN") {
    const int N = 4;
    uint16_t bfloat16_bits[N] = {0b0111111111000001,  // qNaN
                                 0b1111111111000001,
                                 0b0111111110000001,  // sNaN
                                 0b1111111110000001};

    for (int i = 0; i < N; i++) {
      CHECK(std::isnan(sycl::bit_cast<bfloat16>(bfloat16_bits[i])));
    }
  }

  SECTION("Infinity") {
    uint16_t bfloat16_bits_0 = 0b0111111110000000;
    uint16_t bfloat16_bits_1 = 0b1111111110000000;

    CHECK(std::isinf(sycl::bit_cast<bfloat16>(bfloat16_bits_0)));
    CHECK(std::isinf(sycl::bit_cast<bfloat16>(bfloat16_bits_1)));
  }
}

template <typename T>
void check_comparison_op_for_type() {
  const bfloat16 bf = 1;
  const T value = 0;

  CHECK_FALSE(bf == value);
  CHECK(bf != value);
  CHECK(value < bf);
  CHECK(bf > value);
  CHECK(value <= bf);
  CHECK(bf >= value);

  CHECK(std::is_same_v<bool, decltype(bf == value)>);
  CHECK(std::is_same_v<bool, decltype(bf != value)>);
  CHECK(std::is_same_v<bool, decltype(value < bf)>);
  CHECK(std::is_same_v<bool, decltype(bf > value)>);
  CHECK(std::is_same_v<bool, decltype(value <= bf)>);
  CHECK(std::is_same_v<bool, decltype(bf >= value)>);
}

TEST_CASE("Operators", "[bfloat16]") {
  const bfloat16 bf = 2;

  SECTION("operator -") {
    bfloat16 neg_bf = -bf;

    CHECK(neg_bf == -bf);
    CHECK(bf == -neg_bf);
  }

  SECTION("Increment and decrement") {
    // Prefix
    {
      bfloat16 bf1 = 1;
      bfloat16 bf2 = 2;
      auto bf_inc = ++bf1;
      auto bf_dec = --bf2;

      CHECK(bf1 == 2);
      CHECK(bf2 == 1);
      CHECK(bf_inc == 2);
      CHECK(bf_dec == 1);
      CHECK(std::is_same_v<bfloat16&, decltype(++bf1)>);
      CHECK(std::is_same_v<bfloat16&, decltype(--bf2)>);
      CHECK(std::is_same_v<bfloat16, decltype(bf_inc)>);
      CHECK(std::is_same_v<bfloat16, decltype(bf_dec)>);
    }
    // Postfix
    {
      bfloat16 bf1 = 1;
      bfloat16 bf2 = 2;
      auto bf_inc = bf1++;
      auto bf_dec = bf2--;

      CHECK(bf1 == 2);
      CHECK(bf2 == 1);
      CHECK(bf_inc == 1);
      CHECK(bf_dec == 2);
      CHECK(std::is_same_v<bfloat16, decltype(bf1++)>);
      CHECK(std::is_same_v<bfloat16, decltype(bf2--)>);
      CHECK(std::is_same_v<bfloat16, decltype(bf_inc)>);
      CHECK(std::is_same_v<bfloat16, decltype(bf_dec)>);
    }
  }

  SECTION("Assignment operators") {
    bfloat16 bf1 = 1;
    bfloat16 bf2 = 2;
    bfloat16 bf3 = 3;
    bfloat16 bf4 = 4;

    bf1 += bf;
    bf2 -= bf;
    bf3 *= bf;
    bf4 /= bf;

    CHECK(bf1 == 3);
    CHECK(bf2 == 0);
    CHECK(bf3 == 6);
    CHECK(bf4 == 2);

    CHECK(std::is_same_v<decltype(bf1 += bf), bfloat16&>);
    CHECK(std::is_same_v<decltype(bf2 -= bf), bfloat16&>);
    CHECK(std::is_same_v<decltype(bf3 *= bf), bfloat16&>);
    CHECK(std::is_same_v<decltype(bf4 /= bf), bfloat16&>);
  }

  SECTION("Arithmetic operators") {
    bfloat16 bf1 = 1;
    bfloat16 bf2 = 2;
    bfloat16 bf3 = 3;
    bfloat16 bf4 = 4;

    CHECK(bf1 + bf == 3);
    CHECK(bf2 - bf == 0);
    CHECK(bf3 * bf == 6);
    CHECK(bf4 / bf == 2);

    CHECK(std::is_same_v<decltype(bf1 + bf), bfloat16>);
    CHECK(std::is_same_v<decltype(bf2 - bf), bfloat16>);
    CHECK(std::is_same_v<decltype(bf3 * bf), bfloat16>);
    CHECK(std::is_same_v<decltype(bf4 / bf), bfloat16>);
  }

  SECTION("Comparison operators between two bfloat16") {
    check_comparison_op_for_type<bfloat16>();
  }

  SECTION("Comparison operators between bfloat16 and T") {
    check_comparison_op_for_type<char>();
    check_comparison_op_for_type<short>();
    check_comparison_op_for_type<int>();
    check_comparison_op_for_type<long long>();
    check_comparison_op_for_type<std::size_t>();
    check_comparison_op_for_type<bool>();
    check_comparison_op_for_type<float>();

    auto device = sycl_cts::util::get_cts_object::device();
    if (device.has(sycl::aspect::fp16)) {
      check_comparison_op_for_type<sycl::half>();
    }
    if (device.has(sycl::aspect::fp64)) {
      check_comparison_op_for_type<double>();
    }
  }
}

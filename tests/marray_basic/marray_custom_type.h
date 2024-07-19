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

#ifndef SYCL_CTS_TEST_MARRAY_MARRAY_CUSTOM_TYPE_H
#define SYCL_CTS_TEST_MARRAY_MARRAY_CUSTOM_TYPE_H

#include <cstdint>
#include <type_traits>

/// Custom type that meets the NumericType requirement.
struct custom_int {
  custom_int() = default;
  constexpr custom_int(int i) : x{i} {};
  operator int() { return x; }
  custom_int(const custom_int& other) = default;
  custom_int(custom_int&& other) noexcept = default;

  custom_int& operator=(const custom_int& other) = default;

  custom_int& operator=(custom_int&& other) noexcept = default;

  custom_int& operator=(const int& val) {
    *this = custom_int(val);
    return *this;
  }

  custom_int& operator+=(const custom_int& val) {
    x += val.x;
    return *this;
  }
  custom_int& operator-=(const custom_int& val) {
    x -= val.x;
    return *this;
  }
  custom_int& operator*=(const custom_int& val) {
    x *= val.x;
    return *this;
  }
  custom_int& operator/=(const custom_int& val) {
    x /= val.x;
    return *this;
  }
  custom_int& operator%=(const custom_int& val) {
    x %= val.x;
    return *this;
  }
  custom_int& operator&=(const custom_int& val) {
    x &= val.x;
    return *this;
  }
  custom_int& operator|=(const custom_int& val) {
    x |= val.x;
    return *this;
  }
  custom_int& operator^=(const custom_int& val) {
    x ^= val.x;
    return *this;
  }
  custom_int& operator<<=(const custom_int& val) {
    x <<= val.x;
    return *this;
  }
  custom_int& operator>>=(const custom_int& val) {
    x >>= val.x;
    return *this;
  }

  custom_int operator+() const { return +x; }

  custom_int operator-() const { return -x; }

  custom_int operator~() const { return ~x; }

  bool operator!() const { return !x; }

  custom_int& operator++() {
    ++x;
    return *this;
  }

  custom_int operator++(int) {
    custom_int tmp(*this);
    operator++();
    return tmp;
  }

  custom_int& operator--() {
    --x;
    return *this;
  }

  custom_int operator--(int) {
    custom_int tmp(*this);
    operator--();
    return tmp;
  }

  friend custom_int operator+(custom_int lhs, const custom_int& rhs) {
    lhs += rhs;
    return lhs;
  }
  friend custom_int operator-(custom_int lhs, const custom_int& rhs) {
    lhs -= rhs;
    return lhs;
  }
  friend custom_int operator*(custom_int lhs, const custom_int& rhs) {
    lhs *= rhs;
    return lhs;
  }
  friend custom_int operator/(custom_int lhs, const custom_int& rhs) {
    lhs /= rhs;
    return lhs;
  }
  friend custom_int operator%(custom_int lhs, const custom_int& rhs) {
    lhs %= rhs;
    return lhs;
  }
  friend custom_int operator&(custom_int lhs, const custom_int& rhs) {
    lhs &= rhs;
    return lhs;
  }
  friend custom_int operator|(custom_int lhs, const custom_int& rhs) {
    lhs |= rhs;
    return lhs;
  }
  friend custom_int operator^(custom_int lhs, const custom_int& rhs) {
    lhs ^= rhs;
    return lhs;
  }
  friend custom_int operator<<(custom_int lhs, const custom_int& rhs) {
    lhs <<= rhs;
    return lhs;
  }
  friend custom_int operator>>(custom_int lhs, const custom_int& rhs) {
    lhs >>= rhs;
    return lhs;
  }
  friend bool operator&&(custom_int lhs, const custom_int& rhs) {
    return (!!lhs) && (!!rhs);
  }
  friend bool operator||(custom_int lhs, const custom_int& rhs) {
    return (!!lhs) || (!!rhs);
  }

  friend bool operator==(const custom_int& lhs, const custom_int& rhs) {
    return lhs.x == rhs.x;
  }
  friend bool operator!=(const custom_int& lhs, const custom_int& rhs) {
    return !(lhs == rhs);
  }
  friend bool operator<(const custom_int& lhs, const custom_int& rhs) {
    return lhs.x < rhs.x;
  }
  friend bool operator>(const custom_int& lhs, const custom_int& rhs) {
    return lhs.x > rhs.x;
  }
  friend bool operator<=(const custom_int& lhs, const custom_int& rhs) {
    return !(lhs > rhs);
  }
  friend bool operator>=(const custom_int& lhs, const custom_int& rhs) {
    return !(lhs < rhs);
  }

  int x;
};

static_assert(std::is_default_constructible_v<custom_int>);
static_assert(std::is_copy_constructible_v<custom_int>);
static_assert(std::is_copy_assignable_v<custom_int>);
static_assert(std::is_destructible_v<custom_int>);

#endif  // SYCL_CTS_TEST_MARRAY_MARRAY_CUSTOM_TYPE_H

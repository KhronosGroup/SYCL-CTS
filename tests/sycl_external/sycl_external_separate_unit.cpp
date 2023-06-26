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

#include <sycl/sycl.hpp>

constexpr int value = 42;

using accT = sycl::accessor<int, 1>;
using hostAccT = sycl::host_accessor<int, 1>;

#ifdef SYCL_EXTERNAL

SYCL_EXTERNAL void simple_separate_unit_device(const accT& acc) {
  acc[0] = value;
}
SYCL_EXTERNAL void simple_separate_unit_host(hostAccT& acc) { acc[0] = value; }

SYCL_EXTERNAL void extern_separate_unit_device(const accT& acc) {
  acc[0] = value;
}
SYCL_EXTERNAL void extern_separate_unit_host(hostAccT& acc) { acc[0] = value; }

template <sycl::aspect aspect>
SYCL_EXTERNAL [[sycl::device_has(aspect)]] void before_aspect_device(
    const accT& acc);
template <>
SYCL_EXTERNAL void before_aspect_device<sycl::aspect::cpu>(const accT& acc) {
  acc[0] = value;
}
template <>
SYCL_EXTERNAL void before_aspect_device<sycl::aspect::gpu>(const accT& acc) {
  acc[0] = value;
}
template <>
SYCL_EXTERNAL void before_aspect_device<sycl::aspect::accelerator>(
    const accT& acc) {
  acc[0] = value;
}
template <sycl::aspect aspect>
SYCL_EXTERNAL [[sycl::device_has(aspect)]] void before_aspect_host(
    hostAccT& acc);
template <>
SYCL_EXTERNAL void before_aspect_host<sycl::aspect::cpu>(hostAccT& acc) {
  acc[0] = value;
}
template <>
SYCL_EXTERNAL void before_aspect_host<sycl::aspect::gpu>(hostAccT& acc) {
  acc[0] = value;
}
template <>
SYCL_EXTERNAL void before_aspect_host<sycl::aspect::accelerator>(
    hostAccT& acc) {
  acc[0] = value;
}

template <sycl::aspect aspect>
[[sycl::device_has(aspect)]] SYCL_EXTERNAL [[noreturn]] void
between_aspects_device(const accT& acc);
template <>
SYCL_EXTERNAL void between_aspects_device<sycl::aspect::cpu>(const accT& acc) {
  acc[0] = value;
}
template <>
SYCL_EXTERNAL void between_aspects_device<sycl::aspect::gpu>(const accT& acc) {
  acc[0] = value;
}
template <>
SYCL_EXTERNAL void between_aspects_device<sycl::aspect::accelerator>(
    const accT& acc) {
  acc[0] = value;
}
template <sycl::aspect aspect>
[[sycl::device_has(aspect)]] SYCL_EXTERNAL [[noreturn]] void
between_aspects_host(hostAccT& acc);
template <>
SYCL_EXTERNAL void between_aspects_host<sycl::aspect::cpu>(hostAccT& acc) {
  acc[0] = value;
}
template <>
SYCL_EXTERNAL void between_aspects_host<sycl::aspect::gpu>(hostAccT& acc) {
  acc[0] = value;
}
template <>
SYCL_EXTERNAL void between_aspects_host<sycl::aspect::accelerator>(
    hostAccT& acc) {
  acc[0] = value;
}

template <sycl::aspect aspect>
[[sycl::device_has(aspect)]] SYCL_EXTERNAL void after_aspect_device(
    const accT& acc);
template <>
SYCL_EXTERNAL void after_aspect_device<sycl::aspect::cpu>(const accT& acc) {
  acc[0] = value;
}
template <>
SYCL_EXTERNAL void after_aspect_device<sycl::aspect::gpu>(const accT& acc) {
  acc[0] = value;
}
template <>
SYCL_EXTERNAL void after_aspect_device<sycl::aspect::accelerator>(
    const accT& acc) {
  acc[0] = value;
}
template <sycl::aspect aspect>
[[sycl::device_has(aspect)]] SYCL_EXTERNAL void after_aspect_host(
    hostAccT& acc);
template <>
SYCL_EXTERNAL void after_aspect_host<sycl::aspect::cpu>(hostAccT& acc) {
  acc[0] = value;
}
template <>
SYCL_EXTERNAL void after_aspect_host<sycl::aspect::gpu>(hostAccT& acc) {
  acc[0] = value;
}
template <>
SYCL_EXTERNAL void after_aspect_host<sycl::aspect::accelerator>(hostAccT& acc) {
  acc[0] = value;
}
#endif  // SYCL_EXTERNAL

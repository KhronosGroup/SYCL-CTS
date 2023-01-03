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
#include "../common/semantics_reference.h"

template <int Dimensions>
struct storage {
  std::size_t byte_size;
  std::size_t size;
  sycl::range<Dimensions> range =
      sycl_cts::util::get_cts_object::range<Dimensions>::get(1, 1, 1);
  bool is_sub_buffer;

  template <typename T>
  explicit storage(const sycl::buffer<T, Dimensions>& buffer)
      : byte_size(buffer.byte_size()),
        size(buffer.size()),
        range(buffer.get_range()),
        is_sub_buffer(buffer.is_sub_buffer()) {}

  template <typename T>
  bool check(const sycl::buffer<T, Dimensions>& buffer) const {
    return buffer.byte_size() == byte_size && buffer.size() == size &&
           buffer.get_range() == range &&
           buffer.is_sub_buffer() == is_sub_buffer;
  }
};

TEST_CASE("buffer common reference semantics", "[buffer]") {
  sycl::buffer<int> buffer_0{sycl::range<1>{1}};
  sycl::buffer<int> buffer_1{sycl::range<1>{1}};

  common_reference_semantics::check_host<storage<1>>(buffer_0, buffer_1,
                                                     "buffer<int>");
}

TEST_CASE("buffer common reference semantics, mutation", "[buffer]") {
  constexpr int val = 1;
  constexpr int new_val = 2;
  sycl::buffer<int> t0{sycl::range<1>{1}};

  SECTION("mutation to copy") {
    t0.get_host_access()[0] = val;
    sycl::buffer<int> t1(t0);
    t1.get_host_access()[0] = new_val;
    CHECK(new_val == t1.get_host_access()[0]);
  }

  SECTION("mutation to original") {
    t0.get_host_access()[0] = val;
    sycl::buffer<int> t1(t0);
    t0.get_host_access()[0] = new_val;
    CHECK(new_val == t1.get_host_access()[0]);
  }

  // mutation to original, const copy
  // Not possible, since accessor cannot be constructed with a const buffer,
  // and hence its contents cannot be verified.
}

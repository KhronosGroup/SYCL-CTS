/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
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

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
//  Provides common methods for memcpy2d tests
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_EXTENSION_MEMCPY2D_COMMON_H
#define __SYCLCTS_TESTS_EXTENSION_MEMCPY2D_COMMON_H

#include "../../common/common.h"
#include "../../common/type_coverage.h"

namespace memcpy2d_common_tests {

constexpr int expected_val = 42;
constexpr int init_val = 1;

// Width and height of copied and filled regions
constexpr size_t region_width = 5;
constexpr size_t region_height = 8;

// src array width
constexpr size_t src_pitch = 10;

// dest array width
constexpr size_t dest_pitch = 12;

// src and dest arrays height
constexpr size_t array_height = 32;

// Down shift of copied and filled regions
constexpr size_t shift_row = 2;

// Right shift of copied and filled regions
constexpr size_t shift_col = 3;

// enum covers four types of pointers used in the tests
enum class pointer_type {
  host = 0,        // pointer to host dynamic memory
  usm_host = 1,    // pointer to host USM
  usm_device = 2,  // pointer to device USM
  usm_shared = 3   // pointer to shared USM
};

// struct custom_type covers check of trivially copyable custom type
struct custom_type {
  int m_int_field{};
  bool m_bool_field{};

  custom_type() = default;

  custom_type(int value) {
    m_int_field = value;
    m_bool_field = value;
  }

  friend bool operator==(const custom_type& c_t_l, const custom_type& c_t_r) {
    return c_t_l.m_int_field == c_t_r.m_int_field &&
           c_t_l.m_bool_field == c_t_r.m_bool_field;
  }

  operator int() const { return m_int_field; }
};

/**
 * @brief Factory function for getting type_pack with pointer types
 */
inline auto get_pointer_types() {
  static const auto pointer_types =
      value_pack<pointer_type, pointer_type::host, pointer_type::usm_host,
                 pointer_type::usm_device,
                 pointer_type::usm_shared>::generate_named();
  return pointer_types;
}

/**
 * @brief Factory function for getting type_pack with generic types
 */
inline auto get_conformance_type_pack() {
  static const auto types = named_type_pack<
      char, unsigned char, short, unsigned short, int, unsigned int, long,
      unsigned long, long long, unsigned long long, std::size_t, bool,
      custom_type>::generate("char", "unsigned char", "short", "unsigned short",
                             "int", "unsigned int", "long", "unsigned long",
                             "long long", "unsigned long long", "std::size_t",
                             "bool", "custom_type");
  return types;
}

/**
 * @brief Function to check if the device can allocate memory according to the
 * PtrType pointer type
 */
template <pointer_type PtrType>
bool check_device_aspect_allocations(sycl::queue& queue) {
  switch (PtrType) {
    case pointer_type::host:
      return true;
    case pointer_type::usm_host:
      return queue.get_device().has(sycl::aspect::usm_host_allocations);
    case pointer_type::usm_device:
      return queue.get_device().has(sycl::aspect::usm_device_allocations);
    case pointer_type::usm_shared:
      return queue.get_device().has(sycl::aspect::usm_shared_allocations);
  }
}

/**
 * @brief Function to check if the device can allocate memory according to the
 * SrcPtrType and DestPtrType pointer types
 */
template <pointer_type SrcPtrType, pointer_type DestPtrType>
bool check_device_aspect_allocations(sycl::queue& queue) {
  return check_device_aspect_allocations<SrcPtrType>(queue) &&
         check_device_aspect_allocations<DestPtrType>(queue);
}

/**
 * @brief Function to allocate memory with type according to PtrType
 */
template <typename T, pointer_type PtrType>
auto allocate_memory(size_t size, sycl::queue& queue) {
  if constexpr (pointer_type::host == PtrType) {
    return std::make_unique<T[]>(size);
  } else {
    auto deleter = [=](T* ptr) { sycl::free(ptr, queue); };
    if constexpr (pointer_type::usm_host == PtrType) {
      return std::unique_ptr<T[], decltype(deleter)>(
          sycl::malloc_host<T>(size, queue), deleter);
    } else if constexpr (pointer_type::usm_device == PtrType) {
      return std::unique_ptr<T[], decltype(deleter)>(
          sycl::malloc_device<T>(size, queue), deleter);
    } else {
      return std::unique_ptr<T[], decltype(deleter)>(
          sycl::malloc_shared<T>(size, queue), deleter);
    }
  }
}

/**
 * @brief Function to fill memory with type according to PtrType
 */
template <typename T, pointer_type PtrType>
void fill_memory(T* ptr, const T& pattern, size_t count, sycl::queue& queue) {
  if constexpr (PtrType == pointer_type::usm_device) {
    auto event = queue.fill(ptr, pattern, count);
    event.wait();
  } else {
    std::fill(ptr, ptr + count, pattern);
  }
}

/**
 * @brief Function to copy data from memory with type according with PtrType to
 * result array on host
 */
template <pointer_type DestPtrType, typename T>
void copy_destination_to_host_result(T* src, T* result, size_t size,
                                     sycl::queue& queue) {
  if constexpr (DestPtrType == pointer_type::usm_device) {
    queue.copy(src, result, size);
    queue.wait();
  } else {
    std::copy(src, src + size, result);
  }
}

/**
 * @brief Function to get expected value in the allocated memory cell
 */
template <typename T>
T get_expected_value(size_t address, T& ival, T& chval) {
  size_t first_row = shift_row;
  size_t last_row = shift_row + region_height;
  size_t first_col = shift_col;
  size_t last_col = shift_col + region_width;
  size_t row = address / dest_pitch;
  size_t col = address % dest_pitch;
  return (row >= first_row && row < last_row && col >= first_col &&
          col < last_col)
             ? chval
             : ival;
}

/**
 * @brief Function to apply functor for each cell in array
 */
template <typename Functor>
void for_index(Functor&& functor) {
  for (size_t i = 0; i < array_height; ++i) {
    for (size_t j = 0; j < dest_pitch; ++j) {
      size_t index = j + dest_pitch * i;
      functor(index);
    }
  }
}

/**
 * @brief Function for calculating address in array depends on pitch size
 */
template <typename T>
T* get_region_address(T* base, size_t pitch) {
  return base + shift_row * pitch + shift_col;
};

}  // namespace memcpy2d_common_tests

#endif  // __SYCLCTS_TESTS_EXTENSION_MEMCPY2D_COMMON_H

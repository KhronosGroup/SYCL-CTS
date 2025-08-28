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

namespace static_addrspace_cast::tests {

TEST_CASE(
    "the static addrspace cast extension defines the "
    "SYCL_KHR_STATIC_ADDRSPACE_CAST macro",
    "[khr_static_addrspace_cast]") {
#ifndef SYCL_KHR_STATIC_ADDRSPACE_CAST
  static_assert(false, "SYCL_KHR_STATIC_ADDRSPACE_CAST is not defined");
#endif
}

/*
 * This function checks the static address space cast functionality
 * by creating a global buffer with the specified dimensions and
 * casting a raw pointer to that buffer to a global address space pointer.
 * It verifies that the casted pointer matches the original raw pointer.
 *
 * @tparam Dims The dimensions of the test kernel.
 */
template <typename T, std::size_t... Dims>
static void check_global_ptr_cast() {
  // Define the dimensions of the global buffer
  constexpr int dimensions = sizeof...(Dims);
  // Define a queue to submit the kernel
  sycl::queue q = sycl_cts::util::get_cts_object::queue();
  // Define the kernel range
  sycl::range<dimensions> range{Dims...};
  // Create a global buffer with the specified dimensions
  sycl::buffer<int, dimensions> global_buffer{range};
  // Create a result buffer to store the test results
  sycl::buffer<bool, dimensions> result_buffer{range};

  // Submit a kernel to the queue
  q.submit([&](sycl::handler& cgh) {
    auto global_acc =
        global_buffer.template get_access<sycl::access::mode::read_write>(cgh);
    auto result_acc =
        result_buffer.template get_access<sycl::access::mode::write>(cgh);
    cgh.parallel_for(range, [=](sycl::item<dimensions> item) {
      // Get a raw pointer to the global buffer
      T raw_global_ptr{
          global_acc.template get_multi_ptr<sycl::access::decorated::no>()};

      // Cast the raw pointer to a global address space pointer
      auto global_ptr = sycl::khr::static_addrspace_cast<
          sycl::access::address_space::global_space>(raw_global_ptr);

      // Check if the casted pointer matches the original raw pointer
      bool success = (sycl::bit_cast<std::size_t>(raw_global_ptr) ==
                      sycl::bit_cast<std::size_t>(global_ptr.get_raw()));

      // Store the result of the check in the result buffer
      result_acc[item.get_id()] = success;
    });
  });

  // Check the results of the test
  auto result_acc = sycl::host_accessor{result_buffer};
  for (const auto& result : result_acc) CHECK(result);
}

TEST_CASE(
    "static_addrspace_cast works on pointers in address_space::global_space",
    "[khr_static_addrspace_cast]") {
  check_global_ptr_cast<int*, 4>();
  check_global_ptr_cast<int*, 2, 3>();
  check_global_ptr_cast<int*, 2, 3, 4>();
  check_global_ptr_cast<sycl::generic_ptr<int>, 4>();
  check_global_ptr_cast<sycl::generic_ptr<int>, 2, 3>();
  check_global_ptr_cast<sycl::generic_ptr<int>, 4, 2, 3>();
  check_global_ptr_cast<sycl::raw_generic_ptr<int>, 4>();
  check_global_ptr_cast<sycl::raw_generic_ptr<int>, 2, 3>();
  check_global_ptr_cast<sycl::raw_generic_ptr<int>, 2, 3, 4>();
  check_global_ptr_cast<sycl::decorated_generic_ptr<int>, 4>();
  check_global_ptr_cast<sycl::decorated_generic_ptr<int>, 2, 3>();
  check_global_ptr_cast<sycl::decorated_generic_ptr<int>, 2, 3, 4>();
}

/*
 * This function checks the static address space cast functionality by creating
 * a local accessor and casting a raw pointer to that accessor to a local
 * address space pointer. It verifies that the casted pointer matches the
 * original raw pointer.
 *
 * @tparam Dims The dimensions of the test kernel.
 */
template <typename T, std::size_t... Dims>
static void check_local_ptr_cast() {
  // Define the dimensions of the local buffer
  constexpr int dimensions = sizeof...(Dims);
  // Define a queue to submit the kernel
  sycl::queue q = sycl_cts::util::get_cts_object::queue();
  // Define the kernel range
  sycl::range<dimensions> range{Dims...};
  // Create a result buffer to store the test results
  sycl::buffer<bool, dimensions> result_buffer{range};

  // Submit a kernel to the queue
  q.submit([&](sycl::handler& cgh) {
    auto result_acc =
        result_buffer.template get_access<sycl::access::mode::write>(cgh);
    auto local_acc = sycl::local_accessor<int, 1>{1, cgh};
    cgh.parallel_for(
        sycl::nd_range<dimensions>{range, sycl::range<dimensions>{} + 1},
        [=](sycl::nd_item<dimensions> item) {
          // Get a raw pointer to the local accessor
          T raw_local_ptr{
              local_acc.get_multi_ptr<sycl::access::decorated::no>()};

          // Cast the raw pointer to a local address space pointer
          auto local_ptr = sycl::khr::static_addrspace_cast<
              sycl::access::address_space::local_space>(raw_local_ptr);

          // Check if the casted pointer matches the original raw pointer
          bool success = (sycl::bit_cast<std::size_t>(raw_local_ptr) ==
                          sycl::bit_cast<std::size_t>(local_ptr.get_raw()));

          // Store the result of the check in the result buffer
          result_acc[item.get_global_id()] = success;
        });
  });

  // Check the results of the test
  auto result_acc = sycl::host_accessor{result_buffer};
  for (const auto& result : result_acc) CHECK(result);
}

TEST_CASE(
    "static_addrspace_cast works on pointers in address_space::local_space",
    "[khr_static_addrspace_cast]") {
  check_local_ptr_cast<int*, 4>();
  check_local_ptr_cast<int*, 2, 3>();
  check_local_ptr_cast<int*, 2, 3, 4>();
  check_local_ptr_cast<sycl::generic_ptr<int>, 4>();
  check_local_ptr_cast<sycl::generic_ptr<int>, 2, 3>();
  check_local_ptr_cast<sycl::generic_ptr<int>, 4, 2, 3>();
  check_local_ptr_cast<sycl::raw_generic_ptr<int>, 4>();
  check_local_ptr_cast<sycl::raw_generic_ptr<int>, 2, 3>();
  check_local_ptr_cast<sycl::raw_generic_ptr<int>, 2, 3, 4>();
  check_local_ptr_cast<sycl::decorated_generic_ptr<int>, 4>();
  check_local_ptr_cast<sycl::decorated_generic_ptr<int>, 2, 3>();
  check_local_ptr_cast<sycl::decorated_generic_ptr<int>, 2, 3, 4>();
}

/*
 * This function checks the static address space cast functionality
 * by creating a private variable and casting a raw pointer to that variable
 * to a private address space pointer. It verifies that the casted pointer
 * matches the original raw pointer.
 *
 * @tparam Dims The dimensions of the test kernel.
 */
template <typename T, std::size_t... Dims>
static void check_private_ptr_cast() {
  // Define the dimensions of the private buffer
  constexpr int dimensions = sizeof...(Dims);
  // Define a queue to submit the kernel
  sycl::queue q = sycl_cts::util::get_cts_object::queue();
  // Define the kernel range
  sycl::range<dimensions> range{Dims...};
  // Create a result buffer to store the test results
  sycl::buffer<bool, dimensions> result_buffer{range};

  // Submit a kernel to the queue
  q.submit([&](sycl::handler& cgh) {
    auto result_acc =
        result_buffer.template get_access<sycl::access::mode::write>(cgh);
    cgh.parallel_for(range, [=](sycl::item<dimensions> item) {
      // Create a raw pointer to a private variable
      int private_var = 0;
      T raw_private_ptr{&private_var};

      // Cast the raw pointer to a private address space pointer
      auto private_ptr = sycl::khr::static_addrspace_cast<
          sycl::access::address_space::private_space>(raw_private_ptr);

      // Check if the casted pointer matches the original raw pointer
      bool success = (sycl::bit_cast<std::size_t>(raw_private_ptr) ==
                      sycl::bit_cast<std::size_t>(private_ptr.get_raw()));

      // Store the result of the check in the result buffer
      result_acc[item.get_id()] = success;
    });
  });

  // Check the results of the test
  auto result_acc = sycl::host_accessor{result_buffer};
  for (const auto& result : result_acc) CHECK(result);
}

TEST_CASE(
    "static_addrspace_cast works on pointers in address_space::private_space",
    "[khr_static_addrspace_cast]") {
  check_private_ptr_cast<int*, 4>();
  check_private_ptr_cast<int*, 2, 3>();
  check_private_ptr_cast<int*, 2, 3, 4>();
  check_private_ptr_cast<sycl::generic_ptr<int>, 4>();
  check_private_ptr_cast<sycl::generic_ptr<int>, 2, 3>();
  check_private_ptr_cast<sycl::generic_ptr<int>, 4, 2, 3>();
  check_private_ptr_cast<sycl::raw_generic_ptr<int>, 4>();
  check_private_ptr_cast<sycl::raw_generic_ptr<int>, 2, 3>();
  check_private_ptr_cast<sycl::raw_generic_ptr<int>, 2, 3, 4>();
  check_private_ptr_cast<sycl::decorated_generic_ptr<int>, 4>();
  check_private_ptr_cast<sycl::decorated_generic_ptr<int>, 2, 3>();
  check_private_ptr_cast<sycl::decorated_generic_ptr<int>, 2, 3, 4>();
}

}  // namespace static_addrspace_cast::tests

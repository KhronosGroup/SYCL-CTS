/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2026 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "../../../util/sycl_exceptions.h"
#include "../../common/common.h"

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <type_traits>

namespace khr_virtual_mem::tests {

struct aligned_virtual_memory_reservation {
  std::uintptr_t reservation_address;
  std::uintptr_t mapping_address;
  std::size_t reservation_size;
};

static std::size_t get_allocation_granularity(const sycl::queue& queue) {
  const auto context = queue.get_context();
  const auto device = queue.get_device();
  const auto context_minimum = sycl::khr::get_mem_granularity(
      context, sycl::khr::granularity_mode::minimum);
  const auto context_recommended = sycl::khr::get_mem_granularity(
      context, sycl::khr::granularity_mode::recommended);
  const auto device_minimum = sycl::khr::get_mem_granularity(
      device, context, sycl::khr::granularity_mode::minimum);
  const auto device_recommended = sycl::khr::get_mem_granularity(
      device, context, sycl::khr::granularity_mode::recommended);
  REQUIRE(context_minimum > 0);
  REQUIRE(context_recommended > 0);
  REQUIRE(device_minimum > 0);
  REQUIRE(device_recommended > 0);
  return std::lcm(sizeof(int),
                  std::lcm(std::lcm(context_minimum, context_recommended),
                           std::lcm(device_minimum, device_recommended)));
}

static std::size_t align_size(std::size_t value, std::size_t alignment) {
  return value + (alignment - value % alignment) % alignment;
}

static std::uintptr_t align_address(std::uintptr_t value,
                                    std::size_t alignment) {
  return value + (alignment - value % alignment) % alignment;
}

static aligned_virtual_memory_reservation reserve_aligned_virtual_memory(
    const sycl::device& device, const sycl::context& context,
    std::size_t mapping_size) {
  const std::size_t context_granularity = sycl::khr::get_mem_granularity(
      context, sycl::khr::granularity_mode::minimum);
  const std::size_t device_granularity = sycl::khr::get_mem_granularity(
      device, context, sycl::khr::granularity_mode::minimum);
  const std::size_t mapping_alignment =
      std::lcm(device_granularity, alignof(int));
  const std::size_t reservation_size =
      align_size(mapping_size + mapping_alignment, context_granularity);
  const std::uintptr_t reservation =
      sycl::khr::reserve_virtual_mem(reservation_size, context);
  const std::uintptr_t mapping_address =
      align_address(reservation, mapping_alignment);
  return {reservation, mapping_address, reservation_size};
}

static void require_virtual_memory_support(const sycl::queue& queue) {
  for (const auto& device : queue.get_context().get_devices()) {
    if (!device.has(sycl::aspect::khr_virtual_mem))
      SKIP("A device in the context does not support aspect::khr_virtual_mem");
  }
}

TEST_CASE("the virtual memory extension defines the SYCL_KHR_VIRTUAL_MEM macro",
          "[khr_virtual_mem]") {
#ifndef SYCL_KHR_VIRTUAL_MEM
  static_assert(false, "SYCL_KHR_VIRTUAL_MEM is not defined");
#else
  CHECK(SYCL_KHR_VIRTUAL_MEM >= 1);
#endif
}

TEST_CASE("virtual memory granularity queries", "[khr_virtual_mem]") {
  const auto queue = sycl_cts::util::get_cts_object::queue();
  require_virtual_memory_support(queue);
  const auto context = queue.get_context();
  const auto device = queue.get_device();

  const std::size_t context_minimum = sycl::khr::get_mem_granularity(
      context, sycl::khr::granularity_mode::minimum);
  const std::size_t context_recommended = sycl::khr::get_mem_granularity(
      context, sycl::khr::granularity_mode::recommended);
  const std::size_t device_minimum = sycl::khr::get_mem_granularity(
      device, context, sycl::khr::granularity_mode::minimum);
  const std::size_t device_recommended = sycl::khr::get_mem_granularity(
      device, context, sycl::khr::granularity_mode::recommended);

  REQUIRE(context_minimum > 0);
  CHECK(context_recommended >= context_minimum);
  CHECK(context_recommended % context_minimum == 0);
  REQUIRE(device_minimum > 0);
  CHECK(device_recommended >= device_minimum);
  CHECK(device_recommended % device_minimum == 0);
  CHECK(sycl::khr::get_mem_granularity(context) == context_recommended);
  CHECK(sycl::khr::get_mem_granularity(device, context) == device_recommended);
}

TEST_CASE("virtual memory granularity queries reject unsupported devices",
          "[khr_virtual_mem]") {
  const auto queue = sycl_cts::util::get_cts_object::queue();
  const auto device = queue.get_device();
  const auto context = queue.get_context();
  if (device.has(sycl::aspect::khr_virtual_mem))
    SKIP("Device supports aspect::khr_virtual_mem");

  CHECK_THROWS_MATCHES(
      sycl::khr::get_mem_granularity(device, context), sycl::exception,
      sycl_cts::util::equals_exception(sycl::errc::feature_not_supported));
  CHECK_THROWS_MATCHES(
      sycl::khr::get_mem_granularity(context), sycl::exception,
      sycl_cts::util::equals_exception(sycl::errc::feature_not_supported));
}

TEST_CASE("virtual memory reservations accept both overloads",
          "[khr_virtual_mem]") {
  const auto queue = sycl_cts::util::get_cts_object::queue();
  require_virtual_memory_support(queue);
  const auto context = queue.get_context();
  const std::size_t granularity = sycl::khr::get_mem_granularity(
      context, sycl::khr::granularity_mode::minimum);
  REQUIRE(granularity > 0);
  const std::size_t num_bytes = granularity * 2;

  CHECK_THROWS_MATCHES(sycl::khr::reserve_virtual_mem(0, context),
                       sycl::exception,
                       sycl_cts::util::equals_exception(sycl::errc::invalid));
  CHECK_THROWS_MATCHES(sycl::khr::reserve_virtual_mem(0, 0, context),
                       sycl::exception,
                       sycl_cts::util::equals_exception(sycl::errc::invalid));

  const std::uintptr_t first =
      sycl::khr::reserve_virtual_mem(num_bytes, context);
  const std::uintptr_t second =
      sycl::khr::reserve_virtual_mem(first, num_bytes, context);

  CHECK(first % granularity == 0);
  CHECK(second % granularity == 0);
  CHECK(second != first);

  sycl::khr::free_virtual_mem(second, num_bytes, context);
  sycl::khr::free_virtual_mem(first, num_bytes, context);
}

TEST_CASE("physical memory construction and accessors", "[khr_virtual_mem]") {
  static_assert(
      std::is_constructible_v<sycl::khr::physical_mem, sycl::queue, std::size_t,
                              sycl::khr::empty_properties_t>);
  static_assert(!std::is_constructible_v<sycl::khr::physical_mem, sycl::queue,
                                         std::size_t, int>);
  const auto queue = sycl_cts::util::get_cts_object::queue();
  require_virtual_memory_support(queue);
  const std::size_t num_bytes = get_allocation_granularity(queue);
  const auto context = queue.get_context();
  const auto device = queue.get_device();

  CHECK_THROWS_MATCHES(sycl::khr::physical_mem(device, context, 0),
                       sycl::exception,
                       sycl_cts::util::equals_exception(sycl::errc::invalid));
  CHECK_THROWS_MATCHES(sycl::khr::physical_mem(queue, 0), sycl::exception,
                       sycl_cts::util::equals_exception(sycl::errc::invalid));

  sycl::khr::physical_mem physical{device, context, num_bytes};
  CHECK(physical.get_context() == context);
  CHECK(physical.get_device() == device);
  CHECK(physical.size() == num_bytes);

  const std::size_t minimum = sycl::khr::get_mem_granularity(
      device, context, sycl::khr::granularity_mode::minimum);
  REQUIRE(minimum > 0);
  sycl::khr::physical_mem minimum_physical{queue, minimum};
  CHECK(minimum_physical.size() == minimum);
}

TEST_CASE("mapped virtual memory is accessible from a kernel",
          "[khr_virtual_mem]") {
  auto queue = sycl_cts::util::get_cts_object::queue();
  require_virtual_memory_support(queue);
  const auto context = queue.get_context();
  const auto device = queue.get_device();
  const std::size_t mapping_size = get_allocation_granularity(queue);
  const auto [reservation, mapping_address, reservation_size] =
      reserve_aligned_virtual_memory(device, context, mapping_size);
  sycl::khr::physical_mem physical{queue, mapping_size};
  CHECK_THROWS_MATCHES(physical.map(mapping_address, 0,
                                    sycl::khr::address_access_mode::read_write),
                       sycl::exception,
                       sycl_cts::util::equals_exception(sycl::errc::invalid));
  void* mapped = physical.map(mapping_address, mapping_size,
                              sycl::khr::address_access_mode::read_write);

  queue.single_task([=] { static_cast<int*>(mapped)[0] = 1234; })
      .wait_and_throw();
  int result = 0;
  queue.memcpy(&result, mapped, sizeof(result)).wait_and_throw();
  queue.wait_and_throw();

  sycl::khr::unmap(mapped, mapping_size, context);
  sycl::khr::free_virtual_mem(reservation, reservation_size, context);
  CHECK(result == 1234);
}

TEST_CASE("mapped physical memory offsets are honored", "[khr_virtual_mem]") {
  auto queue = sycl_cts::util::get_cts_object::queue();
  require_virtual_memory_support(queue);
  const auto context = queue.get_context();
  const auto device = queue.get_device();
  const std::size_t mapping_size = get_allocation_granularity(queue);
  const auto [reservation, mapping_address, reservation_size] =
      reserve_aligned_virtual_memory(device, context, mapping_size);
  sycl::khr::physical_mem physical{queue, 2 * mapping_size};
  void* mapped =
      physical.map(mapping_address, mapping_size,
                   sycl::khr::address_access_mode::read_write, mapping_size);

  queue.single_task([=] { static_cast<int*>(mapped)[0] = 4321; })
      .wait_and_throw();
  sycl::khr::unmap(mapped, mapping_size, context);

  mapped = physical.map(mapping_address, mapping_size,
                        sycl::khr::address_access_mode::read_write);
  queue.single_task([=] { static_cast<int*>(mapped)[0] = 1234; })
      .wait_and_throw();
  sycl::khr::unmap(mapped, mapping_size, context);

  mapped =
      physical.map(mapping_address, mapping_size,
                   sycl::khr::address_access_mode::read_write, mapping_size);
  int result = 0;
  queue.memcpy(&result, mapped, sizeof(result)).wait_and_throw();
  sycl::khr::unmap(mapped, mapping_size, context);
  sycl::khr::free_virtual_mem(reservation, reservation_size, context);
  CHECK(result == 4321);
}

TEST_CASE("access modes can be queried and changed on subranges",
          "[khr_virtual_mem]") {
  auto queue = sycl_cts::util::get_cts_object::queue();
  require_virtual_memory_support(queue);
  const auto context = queue.get_context();
  const auto device = queue.get_device();
  const std::size_t chunk_size = get_allocation_granularity(queue);
  const std::size_t mapping_size = 3 * chunk_size;
  const auto [reservation, mapping_address, reservation_size] =
      reserve_aligned_virtual_memory(device, context, mapping_size);
  sycl::khr::physical_mem physical{queue, mapping_size};
  void* mapped = physical.map(mapping_address, mapping_size,
                              sycl::khr::address_access_mode::read_write);

  auto* read_only = reinterpret_cast<int*>(mapping_address + chunk_size);
  queue.single_task([=] { read_only[0] = 42; }).wait_and_throw();
  sycl::khr::set_access_mode(mapped, chunk_size,
                             sycl::khr::address_access_mode::none, context);
  sycl::khr::set_access_mode(static_cast<char*>(mapped) + chunk_size,
                             chunk_size, sycl::khr::address_access_mode::read,
                             context);

  const auto first_mode =
      sycl::khr::get_access_mode(mapped, chunk_size, context);
  const auto second_mode = sycl::khr::get_access_mode(
      static_cast<char*>(mapped) + chunk_size, chunk_size, context);
  const auto third_mode = sycl::khr::get_access_mode(
      static_cast<char*>(mapped) + 2 * chunk_size, chunk_size, context);
  int result = 0;
  queue.memcpy(&result, read_only, sizeof(result)).wait_and_throw();

  sycl::khr::unmap(mapped, mapping_size, context);
  sycl::khr::free_virtual_mem(reservation, reservation_size, context);

  CHECK(first_mode == sycl::khr::address_access_mode::none);
  CHECK(second_mode == sycl::khr::address_access_mode::read);
  CHECK(third_mode == sycl::khr::address_access_mode::read_write);
  CHECK(result == 42);
}

TEST_CASE("adjacent mappings extend a stable virtual address range",
          "[khr_virtual_mem]") {
  auto queue = sycl_cts::util::get_cts_object::queue();
  require_virtual_memory_support(queue);
  const auto context = queue.get_context();
  const auto device = queue.get_device();
  const std::size_t chunk_size = get_allocation_granularity(queue);
  const std::size_t mapping_size = 2 * chunk_size;
  const auto [reservation, mapping_address, reservation_size] =
      reserve_aligned_virtual_memory(device, context, mapping_size);
  sycl::khr::physical_mem first_physical{queue, chunk_size};
  sycl::khr::physical_mem second_physical{queue, chunk_size};
  void* first = first_physical.map(mapping_address, chunk_size,
                                   sycl::khr::address_access_mode::read_write);
  queue.single_task([=] { static_cast<int*>(first)[0] = 17; }).wait_and_throw();

  void* second =
      second_physical.map(mapping_address + chunk_size, chunk_size,
                          sycl::khr::address_access_mode::read_write);
  queue.single_task([=] { static_cast<int*>(second)[0] = 29; })
      .wait_and_throw();
  int result[2] = {};
  queue.memcpy(&result[0], first, sizeof(result[0]));
  queue.memcpy(&result[1], second, sizeof(result[1]));
  queue.wait_and_throw();

  sycl::khr::unmap(second, chunk_size, context);
  sycl::khr::unmap(first, chunk_size, context);
  sycl::khr::free_virtual_mem(reservation, reservation_size, context);

  CHECK(result[0] == 17);
  CHECK(result[1] == 29);
  CHECK(reinterpret_cast<std::uintptr_t>(first) == mapping_address);
  CHECK(reinterpret_cast<std::uintptr_t>(second) ==
        mapping_address + chunk_size);
}

TEST_CASE("physical memory contents survive mapping at a new address",
          "[khr_virtual_mem]") {
  auto queue = sycl_cts::util::get_cts_object::queue();
  require_virtual_memory_support(queue);
  const auto context = queue.get_context();
  const auto device = queue.get_device();
  const std::size_t chunk_size = get_allocation_granularity(queue);
  const std::size_t mapping_size = 2 * chunk_size;
  const auto [reservation, mapping_address, reservation_size] =
      reserve_aligned_virtual_memory(device, context, mapping_size);
  sycl::khr::physical_mem physical{queue, chunk_size};
  void* first = physical.map(mapping_address, chunk_size,
                             sycl::khr::address_access_mode::read_write);
  queue.single_task([=] { static_cast<int*>(first)[0] = 55; }).wait_and_throw();
  sycl::khr::unmap(first, chunk_size, context);

  void* second = physical.map(mapping_address + chunk_size, chunk_size,
                              sycl::khr::address_access_mode::read_write);
  int result = 0;
  queue.memcpy(&result, second, sizeof(result)).wait_and_throw();
  queue.wait_and_throw();

  sycl::khr::unmap(second, chunk_size, context);
  sycl::khr::free_virtual_mem(reservation, reservation_size, context);

  CHECK(reinterpret_cast<std::uintptr_t>(first) == mapping_address);
  CHECK(reinterpret_cast<std::uintptr_t>(second) ==
        mapping_address + chunk_size);
  CHECK(result == 55);
}

}  // namespace khr_virtual_mem::tests

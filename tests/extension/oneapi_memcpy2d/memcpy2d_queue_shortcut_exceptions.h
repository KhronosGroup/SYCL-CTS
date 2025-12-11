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
//  Provides class for tests to check queue class functions exceptions gained
//  with oneapi_memcpy2d extension
//
*******************************************************************************/

#ifndef SYCL_CTS_MEMCPY2D_MEMCPY2D_QUEUE_SHORTCUT_EXCEPTIONS_H
#define SYCL_CTS_MEMCPY2D_MEMCPY2D_QUEUE_SHORTCUT_EXCEPTIONS_H

#include <memory>

#include "catch2/catch_test_macros.hpp"

#include "../../../util/sycl_exceptions.h"
#include "../../common/section_name_builder.h"
#include "../../common/value_operations.h"

#include "memcpy2d_common.h"

namespace memcpy2d_queue_shortcut_exceptions_tests {
using namespace memcpy2d_common_tests;

#ifdef SYCL_EXT_ONEAPI_MEMCPY2D

template <typename T, typename SrcPtrT, typename DestPtrT>
class run_queue_shortcut_exceptions_tests {
  static constexpr pointer_type SrcPtrType = SrcPtrT::value;
  static constexpr pointer_type DestPtrType = DestPtrT::value;

 public:
  void operator()(sycl::queue& queue, const std::string& t_name,
                  const std::string& src_ptr_type_name,
                  const std::string& dest_ptr_type_name) {
    // Wrap test code in an outer SECTION to avoid re-executing expensive setup
    // operations (memory allocations, initialization) on every test case
    // re-execution, making the test faster. See the warning in the
    // documentation of `for_all_combinations` for more details.
    SECTION(sycl_cts::section_name(
                std::string("Check queue shortcut exceptions with T = ") +
                t_name + " src_ptr_type = " + src_ptr_type_name +
                " and"
                " dest_ptr_type = " +
                dest_ptr_type_name)
                .create()) {
      run_tests(queue, t_name, src_ptr_type_name, dest_ptr_type_name);
    }
  }

 private:
  void run_tests(sycl::queue& queue, const std::string& t_name,
                 const std::string& src_ptr_type_name,
                 const std::string& dest_ptr_type_name) {
    if (!check_device_aspect_allocations<SrcPtrType, DestPtrType>(queue)) {
      SKIP(
          "Device does not support USM device allocations. "
          "Skipping the test case.");
    }

    T init_v = value_operations::init<T>(init_val);
    T expected_v = value_operations::init<T>(expected_val);

    constexpr size_t extra_bit = 4;

    auto src = allocate_memory<T, SrcPtrType>(
        (src_pitch + extra_bit) * array_height, queue);
    fill_memory<T, SrcPtrType>(src.get(), expected_v,
                               (src_pitch + extra_bit) * array_height, queue);

    auto dst = allocate_memory<T, DestPtrType>(
        (dest_pitch + extra_bit) * array_height, queue);
    fill_memory<T, DestPtrType>(dst.get(), init_v,
                                (dest_pitch + extra_bit) * array_height, queue);

    auto dest_address = get_region_address(dst.get(), dest_pitch);
    auto src_address = get_region_address(src.get(), src_pitch);

    if constexpr (std::is_same_v<T, unsigned char>) {
      SECTION(sycl_cts::section_name(std::string("Check memcpy2d with T = ") +
                                     t_name +
                                     " src_ptr_type = " + src_ptr_type_name +
                                     " and"
                                     " dest_ptr_type = " +
                                     dest_ptr_type_name)
                  .create()) {
        auto action1 = [&](auto reg_width) {
          return queue.ext_oneapi_memcpy2d(dest_address, dest_pitch,
                                           src_address, src_pitch, reg_width,
                                           region_height);
        };
        // Check that if reg_width is greater than src_pitch function throws a
        // synchronous exception with the errc::invalid error code
        CHECK_THROWS_MATCHES(
            action1(src_pitch + extra_bit), sycl::exception,
            sycl_cts::util::equals_exception(sycl::errc::invalid));

        // Check that if reg_width is greater than dest_pitch function throws a
        // synchronous exception with the errc::invalid error code
        CHECK_THROWS_MATCHES(
            action1(dest_pitch + extra_bit), sycl::exception,
            sycl_cts::util::equals_exception(sycl::errc::invalid));

        auto event1 = action1(region_width);
        auto action2 = [&](auto reg_width) {
          return queue.ext_oneapi_memcpy2d(dest_address, dest_pitch,
                                           src_address, src_pitch, reg_width,
                                           region_height, event1);
        };
        // Check that if reg_width is greater than src_pitch function throws a
        // synchronous exception with the errc::invalid error code
        CHECK_THROWS_MATCHES(
            action2(src_pitch + extra_bit), sycl::exception,
            sycl_cts::util::equals_exception(sycl::errc::invalid));

        // Check that if reg_width is greater than dest_pitch function throws a
        // synchronous exception with the errc::invalid error code
        CHECK_THROWS_MATCHES(
            action2(dest_pitch + extra_bit), sycl::exception,
            sycl_cts::util::equals_exception(sycl::errc::invalid));

        auto event2 = action2(region_width);
        auto action3 = [&](auto reg_width) {
          queue.ext_oneapi_memcpy2d(dest_address, dest_pitch, src_address,
                                    src_pitch, reg_width, region_height,
                                    {event1, event2});
        };
        // Check that if reg_width is greater than src_pitch function throws a
        // synchronous exception with the errc::invalid error code
        CHECK_THROWS_MATCHES(
            action3(src_pitch + extra_bit), sycl::exception,
            sycl_cts::util::equals_exception(sycl::errc::invalid));

        // Check that if reg_width is greater than dest_pitch function throws a
        // synchronous exception with the errc::invalid error code
        CHECK_THROWS_MATCHES(
            action3(dest_pitch + extra_bit), sycl::exception,
            sycl_cts::util::equals_exception(sycl::errc::invalid));
        queue.wait_and_throw();
      }
    }
    SECTION(sycl_cts::section_name(std::string("Check copy2d with T = ") +
                                   t_name +
                                   " src_ptr_type = " + src_ptr_type_name +
                                   " and"
                                   " dest_ptr_type = " +
                                   dest_ptr_type_name)
                .create()) {
      auto action1 = [&](auto reg_width) {
        return queue.ext_oneapi_copy2d(src_address, src_pitch, dest_address,
                                       dest_pitch, reg_width, region_height);
      };
      // Check that if reg_width is greater than src_pitch function throws a
      // synchronous exception with the errc::invalid error code
      CHECK_THROWS_MATCHES(
          action1(src_pitch + extra_bit), sycl::exception,
          sycl_cts::util::equals_exception(sycl::errc::invalid));

      // Check that if reg_width is greater than dest_pitch function throws a
      // synchronous exception with the errc::invalid error code
      CHECK_THROWS_MATCHES(
          action1(dest_pitch + extra_bit), sycl::exception,
          sycl_cts::util::equals_exception(sycl::errc::invalid));

      auto event1 = action1(region_width);
      auto action2 = [&](auto reg_width) {
        return queue.ext_oneapi_copy2d(src_address, src_pitch, dest_address,
                                       dest_pitch, reg_width, region_height,
                                       event1);
      };
      // Check that if reg_width is greater than src_pitch function throws a
      // synchronous exception with the errc::invalid error code
      CHECK_THROWS_MATCHES(
          action2(src_pitch + extra_bit), sycl::exception,
          sycl_cts::util::equals_exception(sycl::errc::invalid));

      // Check that if reg_width is greater than dest_pitch function throws a
      // synchronous exception with the errc::invalid error code
      CHECK_THROWS_MATCHES(
          action2(dest_pitch + extra_bit), sycl::exception,
          sycl_cts::util::equals_exception(sycl::errc::invalid));

      auto event2 = action2(region_width);
      auto action3 = [&](auto reg_width) {
        return queue.ext_oneapi_copy2d(src_address, src_pitch, dest_address,
                                       dest_pitch, reg_width, region_height,
                                       {event1, event2});
      };
      // Check that if reg_width is greater than src_pitch function throws a
      // synchronous exception with the errc::invalid error code
      CHECK_THROWS_MATCHES(
          action3(src_pitch + extra_bit), sycl::exception,
          sycl_cts::util::equals_exception(sycl::errc::invalid));

      // Check that if reg_width is greater than dest_pitch function throws a
      // synchronous exception with the errc::invalid error code
      CHECK_THROWS_MATCHES(
          action3(dest_pitch + extra_bit), sycl::exception,
          sycl_cts::util::equals_exception(sycl::errc::invalid));
      queue.wait_and_throw();
    }
    if constexpr (std::is_same_v<T, unsigned char>) {
      SECTION(sycl_cts::section_name(std::string("Check memset2d with T = ") +
                                     t_name +
                                     " src_ptr_type = " + src_ptr_type_name +
                                     " and"
                                     " dest_ptr_type = " +
                                     dest_ptr_type_name)
                  .create()) {
        int value = expected_val;

        auto action1 = [&](auto reg_width) {
          return queue.ext_oneapi_memset2d(dest_address, dest_pitch, value,
                                           reg_width, region_height);
        };
        // Check that if reg_width is greater than dest_pitch function throws a
        // synchronous exception with the errc::invalid error code
        CHECK_THROWS_MATCHES(
            action1(dest_pitch + extra_bit), sycl::exception,
            sycl_cts::util::equals_exception(sycl::errc::invalid));

        auto event1 = action1(region_width);
        auto action2 = [&](auto reg_width) {
          return queue.ext_oneapi_memset2d(dest_address, dest_pitch, value,
                                           reg_width, region_height, event1);
        };
        // Check that if reg_width is greater than dest_pitch function throws a
        // synchronous exception with the errc::invalid error code
        CHECK_THROWS_MATCHES(
            action2(dest_pitch + extra_bit), sycl::exception,
            sycl_cts::util::equals_exception(sycl::errc::invalid));

        auto event2 = action2(region_width);
        auto action3 = [&](auto reg_width) {
          return queue.ext_oneapi_memset2d(dest_address, dest_pitch, value,
                                           reg_width, region_height,
                                           {event1, event2});
        };
        // Check that if reg_width is greater than dest_pitch function throws a
        // synchronous exception with the errc::invalid error code
        CHECK_THROWS_MATCHES(
            action3(dest_pitch + extra_bit), sycl::exception,
            sycl_cts::util::equals_exception(sycl::errc::invalid));
        queue.wait_and_throw();
      }
    }
    SECTION(sycl_cts::section_name(std::string("Check fill2d with T = ") +
                                   t_name +
                                   " src_ptr_type = " + src_ptr_type_name +
                                   " and"
                                   " dest_ptr_type = " +
                                   dest_ptr_type_name)
                .create()) {
      T value = value_operations::init<T>(expected_val);

      auto action1 = [&](auto reg_width) {
        return queue.ext_oneapi_fill2d(dest_address, dest_pitch, value,
                                       reg_width, region_height);
      };
      // Check that if reg_width is greater than dest_pitch function throws a
      // synchronous exception with the errc::invalid error code
      CHECK_THROWS_MATCHES(
          action1(dest_pitch + extra_bit), sycl::exception,
          sycl_cts::util::equals_exception(sycl::errc::invalid));

      auto event1 = action1(region_width);
      auto action2 = [&](auto reg_width) {
        return queue.ext_oneapi_fill2d(dest_address, dest_pitch, value,
                                       reg_width, region_height, event1);
      };
      // Check that if reg_width is greater than dest_pitch function throws a
      // synchronous exception with the errc::invalid error code
      CHECK_THROWS_MATCHES(
          action2(dest_pitch + extra_bit), sycl::exception,
          sycl_cts::util::equals_exception(sycl::errc::invalid));

      auto event2 = action2(region_width);
      auto action3 = [&](auto reg_width) {
        return queue.ext_oneapi_fill2d(dest_address, dest_pitch, value,
                                       reg_width, region_height,
                                       {event1, event2});
      };
      // Check that if reg_width is greater than dest_pitch function throws a
      // synchronous exception with the errc::invalid error code
      CHECK_THROWS_MATCHES(
          action3(dest_pitch + extra_bit), sycl::exception,
          sycl_cts::util::equals_exception(sycl::errc::invalid));
      queue.wait_and_throw();
    }
  }
};

#endif  // SYCL_EXT_ONEAPI_MEMCPY2D

}  // namespace memcpy2d_queue_shortcut_exceptions_tests

#endif  // SYCL_CTS_MEMCPY2D_MEMCPY2D_QUEUE_SHORTCUT_EXCEPTIONS_H

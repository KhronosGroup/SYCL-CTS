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

struct storage {
  bool is_in_order;
  bool has_profiling_enabled;

  static bool device_has_queue_profiling(const sycl::queue& queue) {
    return queue.get_device().has(sycl::aspect::queue_profiling);
  }

  explicit storage(const sycl::queue& queue)
      : is_in_order(queue.is_in_order()) {
    if (device_has_queue_profiling(queue)) {
      has_profiling_enabled =
          queue.has_property<sycl::property::queue::enable_profiling>();
    }
  }

  bool check(const sycl::queue& queue) const {
    return queue.is_in_order() == is_in_order &&
           (!device_has_queue_profiling(queue) ||
            queue.has_property<sycl::property::queue::enable_profiling>() ==
                has_profiling_enabled);
  }
};

TEST_CASE("queue common reference semantics", "[queue]") {
  sycl::queue queue_0{};
  sycl::queue queue_1{};

  common_reference_semantics::check_host<storage>(queue_0, queue_1, "queue");
}

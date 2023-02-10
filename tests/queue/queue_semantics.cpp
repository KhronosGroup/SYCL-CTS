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

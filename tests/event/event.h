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

#ifndef SYCL_CTS_EVENT_EVENT_H
#define SYCL_CTS_EVENT_EVENT_H

#include "../common/common.h"

#include <future>

/**
 * Encapsulates a host task that waits until resolved (= a boolean flag is set).
 */
class resolvable_host_event {
 public:
  /**
   * @param dependencies An optional list of events to depend on.
   */
  resolvable_host_event(const std::vector<sycl::event>& dependencies = {}) {
    event = sycl_cts::util::get_cts_object::queue().submit(
        [this, &dependencies](sycl::handler& cgh) {
          for (auto& dep : dependencies) {
            cgh.depends_on(dep);
          }
          cgh.host_task([this] { promise.get_future().wait(); });
        });
  }

  sycl::event& get_sycl_event() { return event; }

  void resolve() {
    if (!is_resolved) {
      is_resolved = true;
      promise.set_value();
    }
  }

  virtual ~resolvable_host_event() {
    resolve();
    event.wait();
  }

 private:
  bool is_resolved = false;
  std::promise<void> promise;
  sycl::event event;
};

#endif  // SYCL_CTS_EVENT_EVENT_H

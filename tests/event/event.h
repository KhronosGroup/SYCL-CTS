/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
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

/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2022 The Khronos Group Inc.
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

#ifndef SYCL_CTS_QUEUE_QUEUE_SHORTCUTS_USM_H
#define SYCL_CTS_QUEUE_QUEUE_SHORTCUTS_USM_H

#include <utility>

#include "../common/common.h"
#include "../common/get_cts_object.h"
#include "queue_shortcuts_common.h"

namespace queue_shortcuts_usm {

using namespace queue_shortcuts_common;

template <typename T>
struct runner_memcpy {
  runner_memcpy(sycl::queue q, unsigned int count)
      : queue(q), element_count(count) {}

  template <typename kernel_helper>
  sycl::event run_memcpy_copy(kernel_helper helper) {
    // set up initial values on host
    const T t_init{0};
    const T t_test{1};  // to initialize the sequence: t_test, ++t_test, etc.
    std::unique_ptr<T[]> h_src = std::make_unique<T[]>(element_count);
    iota_comp(h_src.get(), h_src.get() + element_count, t_test);
    std::unique_ptr<T[]> h_dest = std::make_unique<T[]>(element_count);
    std::fill(h_dest.get(), h_dest.get() + element_count, t_init);

    // set initial values of device-allocated usm buffer without using shortcuts
    T* d_src = sycl::malloc_device<T>(element_count, queue);
    queue.submit([&](sycl::handler& cgh) {
      cgh.memcpy(d_src, h_src.get(), element_count * sizeof(T));
    });
    T* d_dest = sycl::malloc_device<T>(element_count, queue);
    queue.submit([&](sycl::handler& cgh) {
      cgh.memcpy(d_dest, h_dest.get(), element_count * sizeof(T));
    });
    queue.wait();

    // perform the copy from d_src to d_dest using shortcuts
    sycl::event event = helper(d_dest, d_src);
    event.wait();

    // copy destination buffer back to host without shortcut functions
    std::unique_ptr<T[]> h_actual = std::make_unique<T[]>(element_count);
    queue.submit([&](sycl::handler& cgh) {
      cgh.memcpy(h_actual.get(), d_dest, element_count * sizeof(T));
    });
    queue.wait();

    // check the result
    for (unsigned int i = 0; i < element_count; i++) {
      CHECK((static_cast<T>(t_test + static_cast<int>(i)) == h_actual[i]));
    }

    sycl::free(d_dest, queue);
    sycl::free(d_src, queue);

    return event;
  }

  template <typename kernel_helper>
  sycl::event run_memset(kernel_helper helper) {
    constexpr char init = 0;
    constexpr char test = 1;

    // set up initial values on host
    std::unique_ptr<T[]> h_ptr = std::make_unique<T[]>(element_count);
    std::memset(reinterpret_cast<void*>(h_ptr.get()), init,
                element_count * sizeof(T));

    // set initial values of device-allocated usm buffer without using shortcuts
    T* d_ptr = sycl::malloc_device<T>(element_count, queue);
    queue.submit([&](sycl::handler& cgh) {
      cgh.memcpy(d_ptr, h_ptr.get(), element_count * sizeof(T));
    });
    queue.wait();

    // perform the memset on d_ptr using shortcuts
    sycl::event event = helper(d_ptr, test);
    event.wait();

    // copy the buffer back to host without shortcut functions
    std::unique_ptr<T[]> h_actual = std::make_unique<T[]>(element_count);
    queue.submit([&](sycl::handler& cgh) {
      cgh.memcpy(h_actual.get(), d_ptr, element_count * sizeof(T));
    });
    queue.wait();

    // check the result
    char* actual_char = reinterpret_cast<char*>(h_actual.get());
    for (unsigned int i = 0; i < element_count * sizeof(T); i++) {
      CHECK((test == actual_char[i]));
    }

    sycl::free(d_ptr, queue);

    return event;
  }

  template <typename kernel_helper>
  sycl::event run_fill(kernel_helper helper) {
    // set up initial values on host
    const T t_init{0};
    const T t_test{1};
    std::unique_ptr<T[]> h_ptr = std::make_unique<T[]>(element_count);
    std::fill(h_ptr.get(), h_ptr.get() + element_count, t_init);

    // set initial values of device-allocated usm buffer without using shortcuts
    T* d_ptr = sycl::malloc_device<T>(element_count, queue);
    queue.submit([&](sycl::handler& cgh) {
      cgh.memcpy(d_ptr, h_ptr.get(), element_count * sizeof(T));
    });
    queue.wait();

    // perform the fill on d_ptr using shortcuts
    sycl::event event = helper(d_ptr, t_test);
    event.wait();

    // copy the buffer back to host without shortcut functions
    std::unique_ptr<T[]> h_actual = std::make_unique<T[]>(element_count);
    queue.submit([&](sycl::handler& cgh) {
      cgh.memcpy(h_actual.get(), d_ptr, element_count * sizeof(T));
    });
    queue.wait();

    // check the result
    for (unsigned int i = 0; i < element_count; i++) {
      CHECK((t_test == h_actual[i]));
    }

    sycl::free(d_ptr, queue);

    return event;
  }

  sycl::queue queue;
  const unsigned int element_count;
};

template <typename T>
void test_unified_shared_memory(sycl::queue q, unsigned int element_count) {
  const bool has_usm_device_allocations =
      q.get_device().has(sycl::aspect::usm_device_allocations);
  const bool has_usm_shared_allocations =
      q.get_device().has(sycl::aspect::usm_shared_allocations);

  runner_memcpy<T> runner(q, element_count);

  // memcpy and copy
  if (has_usm_device_allocations) {
    sycl::event memcpy_no_events =
        runner.template run_memcpy_copy<>([&](T* dest, T* src) {
          return q.memcpy(dest, src, element_count * sizeof(T));
        });
    sycl::event memcpy_single_event =
        runner.template run_memcpy_copy<>([&](T* dest, T* src) {
          return q.memcpy(dest, src, element_count * sizeof(T),
                          memcpy_no_events);
        });
    sycl::event memcpy_multiple_events =
        runner.template run_memcpy_copy<>([&](T* dest, T* src) {
          return q.memcpy(dest, src, element_count * sizeof(T),
                          {memcpy_no_events, memcpy_single_event});
        });

    sycl::event copy_no_events = runner.template run_memcpy_copy<>(
        [&](T* dest, T* src) { return q.copy(src, dest, element_count); });
    sycl::event copy_single_event =
        runner.template run_memcpy_copy<>([&](T* dest, T* src) {
          return q.copy(src, dest, element_count, copy_no_events);
        });
    sycl::event copy_multiple_events =
        runner.template run_memcpy_copy<>([&](T* dest, T* src) {
          return q.copy(src, dest, element_count,
                        {copy_no_events, copy_single_event});
        });
  } else {
    WARN(
        "Device does not support USM device allocations. "
        "Skipping the test case.");
  }

  // memset and fill
  if (has_usm_device_allocations) {
    sycl::event memset_no_events =
        runner.template run_memset<>([&](T* ptr, char pattern) {
          return q.memset(ptr, pattern, element_count * sizeof(T));
        });
    sycl::event memset_single_event =
        runner.template run_memset<>([&](T* ptr, char pattern) {
          return q.memset(ptr, pattern, element_count * sizeof(T),
                          memset_no_events);
        });
    sycl::event memset_multiple_events =
        runner.template run_memset<>([&](T* ptr, char pattern) {
          return q.memset(ptr, pattern, element_count * sizeof(T),
                          {memset_no_events, memset_single_event});
        });

    sycl::event fill_no_events = runner.template run_fill<>(
        [&](T* ptr, T pattern) { return q.fill(ptr, pattern, element_count); });
    sycl::event fill_single_event =
        runner.template run_fill<>([&](T* ptr, T pattern) {
          return q.fill(ptr, pattern, element_count, fill_no_events);
        });
    sycl::event fill_multiple_events =
        runner.template run_fill<>([&](T* ptr, T pattern) {
          return q.fill(ptr, pattern, element_count,
                        {fill_no_events, fill_single_event});
        });
  } else {
    WARN(
        "Device does not support USM device allocations. "
        "Skipping the test case.");
  }

  // prefetch
  if (has_usm_shared_allocations) {
    T* ptr = sycl::malloc_shared<T>(element_count, q);
    sycl::event prefetch_no_events = q.prefetch(ptr, element_count * sizeof(T));
    sycl::event prefetch_single_event =
        q.prefetch(ptr, element_count * sizeof(T), prefetch_no_events);
    sycl::event prefetch_multiple_events =
        q.prefetch(ptr, element_count * sizeof(T),
                   {prefetch_no_events, prefetch_single_event});
    prefetch_multiple_events.wait();
    prefetch_no_events.wait();
    sycl::free(ptr, q);
  } else {
    WARN(
        "Device does not support USM shared allocations. "
        "Skipping the test case.");
  }

  // advise
  if (has_usm_device_allocations) {
    T* ptr = sycl::malloc_device<T>(element_count, q);
    constexpr int advice = 0;
    sycl::event advise_no_events =
        q.mem_advise(ptr, element_count * sizeof(int), advice);
    sycl::event advise_single_event = q.mem_advise(
        ptr, element_count * sizeof(int), advice, advise_no_events);
    sycl::event advise_multiple_events =
        q.mem_advise(ptr, element_count * sizeof(int), advice,
                     {advise_no_events, advise_single_event});
    advise_multiple_events.wait();
    advise_no_events.wait();
    sycl::free(ptr, q);
  } else {
    WARN(
        "Device does not support USM device allocations. "
        "Skipping the test case.");
  }
}

template <typename T>
class check_queue_shortcuts_usm_for_type {
  static constexpr unsigned int element_count = 10;

 public:
  void operator()(sycl::queue queue, const std::string& type_name) {
    INFO("for type \"" << type_name << "\": ");

    test_unified_shared_memory<T>(queue, element_count);
  }
};

};  // namespace queue_shortcuts_usm

#endif  // SYCL_CTS_QUEUE_QUEUE_SHORTCUTS_USM_H

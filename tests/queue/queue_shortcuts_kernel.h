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

#ifndef SYCL_CTS_QUEUE_QUEUE_SHORTCUTS_KERNEL_H
#define SYCL_CTS_QUEUE_QUEUE_SHORTCUTS_KERNEL_H

#include "../common/common.h"
#include "../common/disabled_for_test_case.h"
#include "../common/get_cts_object.h"

namespace queue_shortcuts_kernel {

constexpr int value_init = 0;
constexpr int value_test = 1;
static_assert(value_init != value_test);
constexpr int range_dim = 3;

struct simple_functor_single_task {
  explicit simple_functor_single_task(int* p) : usm(p) {}

  void operator()() const { *usm = value_test; }

  int* const usm;
};

struct simple_functor_parallel_for_range {
  explicit simple_functor_parallel_for_range(int* p) : usm(p) {}

  void operator()(sycl::item<range_dim>) const { *usm = value_test; }

  int* const usm;
};

struct simple_functor_parallel_for_nd_range {
  explicit simple_functor_parallel_for_nd_range(int* p) : usm(p) {}

  void operator()(sycl::nd_item<range_dim>) const { *usm = value_test; }

  int* const usm;
};

template <typename T>
class kernel_stnel;
template <typename T>
class kernel_stsel;
template <typename T>
class kernel_stmel;
template <typename T>
class kernel_stnef;
template <typename T>
class kernel_stsef;
template <typename T>
class kernel_stmef;
template <typename T>
class kernel_pfnel;
template <typename T>
class kernel_pfsel;
template <typename T>
class kernel_pfmel;
template <typename T>
class kernel_pfnef;
template <typename T>
class kernel_pfsef;
template <typename T>
class kernel_pfmef;
template <typename T>
class kernel_pfnrnel;
template <typename T>
class kernel_pfnrsel;
template <typename T>
class kernel_pfnrmel;
template <typename T>
class kernel_pfnrnef;
template <typename T>
class kernel_pfnrsef;
template <typename T>
class kernel_pfnrmef;

template <typename kernel_helper>
sycl::event run_test(sycl::queue queue, int* d_ptr, kernel_helper helper) {
  // initialize the device-allocated usm buffer without using shortcuts
  int init = value_init;
  queue.submit(
      [&](sycl::handler& cgh) { cgh.memcpy(d_ptr, &init, sizeof(int)); });
  queue.wait();

  // set the value using shortcuts
  sycl::event event = helper();
  event.wait();

  // copy the value back to host without using shortcuts
  int result;
  queue.submit(
      [&](sycl::handler& cgh) { cgh.memcpy(&result, d_ptr, sizeof(int)); });
  queue.wait();

  CHECK((result == value_test));
  return event;
}

template <typename T>
void test_kernel_function(sycl::queue q) {
  if (!q.get_device().has(sycl::aspect::usm_device_allocations)) {
    WARN(
        "Device does not support USM device allocations. "
        "Skipping the test case.");
    return;
  }

  int* d_ptr = sycl::malloc_device<int>(1, q);

  // single_task
  {
    const auto single_task_lambda = [=] { *d_ptr = value_test; };
    const simple_functor_single_task single_task_functor(d_ptr);

    {
      sycl::event single_task_no_events_lambda = run_test(q, d_ptr, [&] {
        return q.single_task<kernel_stnel<T>>(single_task_lambda);
      });
      sycl::event single_task_single_event_lambda = run_test(q, d_ptr, [&] {
        return q.single_task<kernel_stsel<T>>(single_task_no_events_lambda,
                                              single_task_lambda);
      });
      sycl::event single_task_multiple_events_lambda = run_test(q, d_ptr, [&] {
        return q.single_task<kernel_stmel<T>>(
            {single_task_no_events_lambda, single_task_single_event_lambda},
            single_task_lambda);
      });
    }
    {
      sycl::event single_task_no_events_functor = run_test(q, d_ptr, [&] {
        return q.single_task<kernel_stnef<T>>(single_task_functor);
      });
      sycl::event single_task_single_event_functor = run_test(q, d_ptr, [&] {
        return q.single_task<kernel_stsef<T>>(single_task_no_events_functor,
                                              single_task_functor);
      });
      sycl::event single_task_multiple_events_functor = run_test(q, d_ptr, [&] {
        return q.single_task<kernel_stmef<T>>(
            {single_task_no_events_functor, single_task_single_event_functor},
            single_task_functor);
      });
    }
  }
  // parallel_for range
  {
    const sycl::range<range_dim> range(1, 1, 1);
    const auto parallel_for_lambda = [=](sycl::item<range_dim>) {
      *d_ptr = value_test;
    };
    const simple_functor_parallel_for_range parallel_for_functor(d_ptr);

    {
      sycl::event parallel_for_range_no_events_lambda = run_test(q, d_ptr, [&] {
        return q.parallel_for<kernel_pfnel<T>>(range, parallel_for_lambda);
      });
      sycl::event parallel_for_range_single_event_lambda =
          run_test(q, d_ptr, [&] {
            return q.parallel_for<kernel_pfsel<T>>(
                range, parallel_for_range_no_events_lambda,
                parallel_for_lambda);
          });
      sycl::event parallel_for_range_multiple_events_lambda =
          run_test(q, d_ptr, [&] {
            return q.parallel_for<kernel_pfmel<T>>(
                range,
                {parallel_for_range_no_events_lambda,
                 parallel_for_range_single_event_lambda},
                parallel_for_lambda);
          });
    }
    {
      sycl::event parallel_for_range_no_events_functor =
          run_test(q, d_ptr, [&] {
            return q.parallel_for<kernel_pfnef<T>>(range, parallel_for_functor);
          });
      sycl::event parallel_for_range_single_event_functor =
          run_test(q, d_ptr, [&] {
            return q.parallel_for<kernel_pfsef<T>>(
                range, parallel_for_range_no_events_functor,
                parallel_for_functor);
          });
      sycl::event parallel_for_range_multiple_events_functor =
          run_test(q, d_ptr, [&] {
            return q.parallel_for<kernel_pfmef<T>>(
                range,
                {parallel_for_range_no_events_functor,
                 parallel_for_range_single_event_functor},
                parallel_for_functor);
          });
    }
  }
  // parallel_for nd_range
  {
    const sycl::nd_range<range_dim> range({1, 1, 1}, {1, 1, 1});
    const auto parallel_for_lambda = [=](sycl::nd_item<range_dim>) {
      *d_ptr = value_test;
    };
    const simple_functor_parallel_for_nd_range parallel_for_functor(d_ptr);

    {
      sycl::event parallel_for_range_no_events_lambda = run_test(q, d_ptr, [&] {
        return q.parallel_for<kernel_pfnrnel<T>>(range, parallel_for_lambda);
      });
      sycl::event parallel_for_range_single_event_lambda =
          run_test(q, d_ptr, [&] {
            return q.parallel_for<kernel_pfnrsel<T>>(
                range, parallel_for_range_no_events_lambda,
                parallel_for_lambda);
          });
      sycl::event parallel_for_range_multiple_events_lambda =
          run_test(q, d_ptr, [&] {
            return q.parallel_for<kernel_pfnrmel<T>>(
                range,
                {parallel_for_range_no_events_lambda,
                 parallel_for_range_single_event_lambda},
                parallel_for_lambda);
          });
    }
    {
      sycl::event parallel_for_range_no_events_functor =
          run_test(q, d_ptr, [&] {
            return q.parallel_for<kernel_pfnrnef<T>>(range,
                                                     parallel_for_functor);
          });
      sycl::event parallel_for_range_single_event_functor =
          run_test(q, d_ptr, [&] {
            return q.parallel_for<kernel_pfnrsef<T>>(
                range, parallel_for_range_no_events_functor,
                parallel_for_functor);
          });
      sycl::event parallel_for_range_multiple_events_functor =
          run_test(q, d_ptr, [&] {
            return q.parallel_for<kernel_pfnrmef<T>>(
                range,
                {parallel_for_range_no_events_functor,
                 parallel_for_range_single_event_functor},
                parallel_for_functor);
          });
    }
  }

  sycl::free(d_ptr, q);
}

template <typename T>
class check_queue_shortcuts_kernel_for_type {
 public:
  void operator()(sycl::queue queue, const std::string& type_name) {
    INFO("for type \"" << type_name << "\": ");

    test_kernel_function<T>(queue);
  }
};

}  // namespace queue_shortcuts_kernel

#endif  // SYCL_CTS_QUEUE_QUEUE_SHORTCUTS_KERNEL_H

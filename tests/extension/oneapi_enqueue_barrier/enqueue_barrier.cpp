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
#include "../../common/common.h"

namespace enqueue_barrier::tests {

inline void skip_if_dev_does_not_support_usm_alloc(
    sycl::device& dev, sycl::aspect usm_alloc_type) {
  if (!dev.has(usm_alloc_type)) {
    SKIP("Device doesn't support usm allocations");
  }
}

constexpr size_t buffer_size = 10;
constexpr size_t ker_a_iter_num = 1000000;
constexpr size_t ker_b_iter_num = 3000000;

class EnqueueBarrierTestBase {
  template <size_t iter_num>
  class enqueue_barrier_kernel {
    using LoopArrAccT = sycl::accessor<int, 1, sycl::access_mode::write>;
    using ResBufAccT = sycl::accessor<bool, 1, sycl::access_mode::write>;
    LoopArrAccT loop_acc;
    ResBufAccT res_acc;
    bool* usm_ptr;

   public:
    enqueue_barrier_kernel(LoopArrAccT loop_arr_acc, ResBufAccT res_buf_acc,
                           bool* flags)
        : loop_acc(loop_arr_acc), res_acc(res_buf_acc), usm_ptr(flags){};
    void operator()() const {
      for (size_t i = 0; i < iter_num; i++) {
        int val = sycl::sqrt(float(i));
        loop_acc[val % buffer_size] = i;
      }
      res_acc[0] = usm_ptr[0] == false;
      res_acc[1] = usm_ptr[1] == false;
    }
  };

  virtual void submit_barrier(sycl::event& e1, sycl::event& e2,
                              sycl::queue& q) = 0;

 protected:
  void run(sycl::queue& q1, sycl::queue& q2, sycl::queue& q3) {
    bool* completion_flags = sycl::malloc_device<bool>(2, q1);

    std::array result_in_ker_a{false, false};
    std::array result_in_ker_b{false, false};

    int loop_arr_a[buffer_size];
    int loop_arr_b[buffer_size];

    {
      sycl::buffer<bool, 1> result_buf_a(result_in_ker_a.data(),
                                         result_in_ker_a.size());
      sycl::buffer<bool, 1> result_buf_b(result_in_ker_b.data(),
                                         result_in_ker_b.size());
      sycl::buffer<int, 1> loop_arr_buf_a(loop_arr_a, buffer_size);
      sycl::buffer<int, 1> loop_arr_buf_b(loop_arr_b, buffer_size);

      q1.submit([&](sycl::handler& cgh) {
          cgh.single_task([=] {
            completion_flags[0] = false;
            completion_flags[1] = false;
          });
        }).wait();

      auto event1 = q1.submit([&](sycl::handler& cgh) {
        sycl::accessor result_buf_a_acc(result_buf_a, cgh, sycl::write_only);
        sycl::accessor loop_arr_buf_a_acc(loop_arr_buf_a, cgh, sycl::write_only);
        enqueue_barrier_kernel<ker_a_iter_num> kern_a(
            loop_arr_buf_a_acc, result_buf_a_acc, completion_flags);
        cgh.single_task(kern_a);
      });

      auto event2 = q2.submit([&](sycl::handler& cgh) {
        auto result_buf_b_acc =
            sycl::accessor(result_buf_b, cgh, sycl::write_only);
        auto loop_arr_buf_b_acc =
            sycl::accessor(loop_arr_buf_b, cgh, sycl::write_only);
        enqueue_barrier_kernel<ker_b_iter_num> kern_b(
            loop_arr_buf_b_acc, result_buf_b_acc, completion_flags);
        cgh.single_task(kern_b);
      });

      submit_barrier(event1, event2, q3);

      q3.submit([&](sycl::handler& cgh) {
        cgh.single_task([=] { completion_flags[0] = true; });
      });

      q3.submit([&](sycl::handler& cgh) {
        cgh.single_task([=] { completion_flags[1] = true; });
      });
    }

    sycl::free(completion_flags, q1);

    CHECK(result_in_ker_a[0]);
    CHECK(result_in_ker_a[1]);
    CHECK(result_in_ker_b[0]);
    CHECK(result_in_ker_b[1]);
  }
};

class EnqueueBarrierTestAllPrevEvents : public EnqueueBarrierTestBase {
 public:
  void run() {
    auto queue = sycl_cts::util::get_cts_object::queue();
    EnqueueBarrierTestBase::run(queue, queue, queue);
  }
};

class EnqueueBarrierTestForSpecialEvents : public EnqueueBarrierTestBase {
 public:
  void run() {
    auto queue1 = sycl_cts::util::get_cts_object::queue();

    auto ctxt = queue1.get_context();
    auto dev = queue1.get_device();

    // Using the same context and device to be able to use usm pointer from
    // kernels in other queues
    sycl::queue queue2{ctxt, dev};
    sycl::queue queue3{ctxt, dev};
    EnqueueBarrierTestBase::run(queue1, queue2, queue3);
  }
};

class EnqueueBarrierTestUsingHandler : public EnqueueBarrierTestAllPrevEvents {
  void submit_barrier(sycl::event& e1, sycl::event& e2, sycl::queue& q) {
    q.submit([](sycl::handler& cgh) { cgh.ext_oneapi_barrier(); });
  }
};

class EnqueueBarrierTestUsingHandlerOverloaded
    : public EnqueueBarrierTestForSpecialEvents {
  void submit_barrier(sycl::event& e1, sycl::event& e2, sycl::queue& q) {
    q.submit([&](sycl::handler& cgh) { cgh.ext_oneapi_barrier({e1, e2}); });
  }
};

class EnqueueBarrierTestUsingQueue : public EnqueueBarrierTestAllPrevEvents {
  void submit_barrier(sycl::event& e1, sycl::event& e2, sycl::queue& q) {
    auto ret_value = q.ext_oneapi_submit_barrier();
    STATIC_CHECK(std::is_same_v<decltype(ret_value), sycl::event>);
  }
};

class EnqueueBarrierTestUsingQueueOverloaded
    : public EnqueueBarrierTestForSpecialEvents {
  void submit_barrier(sycl::event& e1, sycl::event& e2, sycl::queue& q) {
    auto ret_value = q.ext_oneapi_submit_barrier({e1, e2});
    STATIC_CHECK(std::is_same_v<decltype(ret_value), sycl::event>);
  }
};

template <typename BarrierFuncT>
void test_returned_event(sycl::queue& queue1, sycl::queue& queue2,
                         sycl::queue& queue3, BarrierFuncT submit_barrier) {
  constexpr int expected_val = 42;
  int* buf_a = sycl::malloc_host<int>(1, queue1);
  int* buf_b = sycl::malloc_host<int>(1, queue1);

  buf_a[0] = 0;
  buf_b[0] = 0;

  int loop_arr_a[buffer_size];
  int loop_arr_b[buffer_size];

  sycl::buffer<int, 1> loop_arr_buf_a(loop_arr_a, buffer_size);
  sycl::buffer<int, 1> loop_arr_buf_b(loop_arr_b, buffer_size);

  auto event1 = queue1.submit([&](sycl::handler& cgh) {
    auto loop_arr_buf_a_acc =
        sycl::accessor(loop_arr_buf_a, cgh, sycl::write_only);
    cgh.single_task([=] {
      for (size_t i = 0; i < ker_a_iter_num; i++) {
        int val = sycl::sqrt(float(i));
        loop_arr_buf_a_acc[val % buffer_size] = i;
      }
      buf_a[0] = expected_val;
    });
  });

  auto event2 = queue2.submit([&](sycl::handler& cgh) {
    auto loop_arr_buf_b_acc =
        sycl::accessor(loop_arr_buf_b, cgh, sycl::write_only);
    cgh.single_task([=] {
      for (size_t i = 0; i < ker_b_iter_num; i++) {
        int val = sycl::sqrt(float(i));
        loop_arr_buf_b_acc[val % buffer_size] = i;
      }
      buf_b[0] = expected_val;
    });
  });

  auto event3 = submit_barrier(event1, event2, queue3);

  event3.wait();

  CHECK(buf_a[0] == expected_val);
  CHECK(buf_b[0] == expected_val);

  sycl::free(buf_a, queue1);
  sycl::free(buf_b, queue1);
}

TEST_CASE("Check sycl_ext_oneapi_enqueue_barrier extension.",
          "[oneapi_enqueue_barrier]") {
  auto queue = sycl_cts::util::get_cts_object::queue();
  auto dev = queue.get_device();

#ifndef SYCL_EXT_ONEAPI_ENQUEUE_BARRIER
  SKIP("SYCL_EXT_ONEAPI_ENQUEUE_BARRIER is not defined");
#else

  SECTION("Check sycl::handler::ext_oneapi_barrier()") {
    skip_if_dev_does_not_support_usm_alloc(
        dev, sycl::aspect::usm_device_allocations);
    EnqueueBarrierTestUsingHandler test;
    test.run();
  }

  SECTION(
      "Check sycl::handler::ext_oneapi_barrier(const std::vector<event> "
      "&waitList)") {
    skip_if_dev_does_not_support_usm_alloc(
        dev, sycl::aspect::usm_device_allocations);
    EnqueueBarrierTestUsingHandlerOverloaded test;
    test.run();
  }

  SECTION("Check sycl::queue::ext_oneapi_submit_barrier()") {
    skip_if_dev_does_not_support_usm_alloc(
        dev, sycl::aspect::usm_device_allocations);
    EnqueueBarrierTestUsingQueue test;
    test.run();
  }

  SECTION(
      "Check sycl::queue::ext_oneapi_submit_barrier(const std::vector<event> "
      "&waitList)") {
    skip_if_dev_does_not_support_usm_alloc(
        dev, sycl::aspect::usm_device_allocations);
    EnqueueBarrierTestUsingQueueOverloaded test;
    test.run();
  }

  SECTION("Check returned event, one queue") {
    skip_if_dev_does_not_support_usm_alloc(dev,
                                           sycl::aspect::usm_host_allocations);
    auto submit_barrier = [](sycl::event& e1, sycl::event& e2, sycl::queue& q) {
      return q.ext_oneapi_submit_barrier();
    };

    test_returned_event(queue, queue, queue, submit_barrier);
  }

  SECTION("Check returned event, three queues") {
    skip_if_dev_does_not_support_usm_alloc(dev,
                                           sycl::aspect::usm_host_allocations);
    auto ctxt = queue.get_context();

    // Using the same context and device to be able to use usm pointer from
    // kernels in other queues
    sycl::queue queue2{ctxt, dev};
    sycl::queue queue3{ctxt, dev};

    auto submit_barrier = [](sycl::event& e1, sycl::event& e2, sycl::queue& q) {
      return q.ext_oneapi_submit_barrier({e1, e2});
    };

    test_returned_event(queue, queue2, queue3, submit_barrier);
  }

#endif  // SYCL_EXT_ONEAPI_ENQUEUE_BARRIER
}

}  // namespace enqueue_barrier::tests

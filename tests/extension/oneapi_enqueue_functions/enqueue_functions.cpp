/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2024 The Khronos Group Inc.
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

namespace enqueue_functions::tests {

#ifdef SYCL_EXT_ONEAPI_ENQUEUE_FUNCTIONS
namespace syclex = sycl::ext::oneapi::experimental;

static void test_single_task() {
  const auto Increment = [](const auto& enqueue,
                            const std::vector<int>& input) {
    const int N = input.size();
    sycl::queue q;
    int* data = sycl::malloc_shared<int>(N, q);
    std::copy(input.begin(), input.end(), data);

    enqueue(q, [=] {
      for (int i = 0; i < N; i++) data[i]++;
    });
    q.wait();

    std::vector<int> output(input.size());
    std::copy(data, data + N, output.begin());
    sycl::free(data, q);
    return output;
  };

  const auto RunKernel = [&](const auto& enqueue) {
    const std::vector<int> input{1, 2, 3, 4};
    return Increment(enqueue, input);
  };

  CHECK(RunKernel([](sycl::queue q, const auto& kernel) {
          q.single_task(kernel);
        }) == RunKernel([](sycl::queue q, const auto& kernel) {
          syclex::single_task(q, kernel);
        }));

  CHECK(RunKernel([](sycl::queue q, const auto& kernel) {
          q.submit([&](sycl::handler& h) { h.single_task(kernel); });
        }) == RunKernel([](sycl::queue q, const auto& kernel) {
          syclex::submit(
              q, [&](sycl::handler& h) { syclex::single_task(h, kernel); });
        }));
}

static void test_parallel_for() {
  const auto Sum = [](const auto& enqueue, const std::vector<int>& input) {
    const int N = input.size();
    sycl::queue q;
    int* data = sycl::malloc_shared<int>(N, q);
    std::copy(input.begin(), input.end(), data);

    int* sum = sycl::malloc_shared<int>(1, q);
    auto reduction = sycl::reduction(sum, std::plus<>());
    sum[0] = 0;

    enqueue(q, sycl::range<1>(N), reduction,
            [=](sycl::id<1> it, auto& sum) { sum.combine(data[it]); });
    q.wait();

    const int output = sum[0];
    sycl::free(sum, q);
    return output;
  };

  const auto RunKernel = [&](const auto& enqueue) {
    const std::vector<int> input{1, 2, 3, 4};
    return Sum(enqueue, input);
  };

  CHECK(RunKernel([](sycl::queue q, sycl::range<1> range, auto reduction,
                     const auto& kernel) {
          q.parallel_for(range, reduction, kernel);
        }) == RunKernel([](sycl::queue q, sycl::range<1> range, auto reduction,
                           const auto& kernel) {
          syclex::parallel_for(q, range, kernel, reduction);
        }));

  CHECK(RunKernel([](sycl::queue q, sycl::range<1> range, auto reduction,
                     const auto& kernel) {
          q.submit([&](sycl::handler& h) {
            h.parallel_for(range, reduction, kernel);
          });
        }) == RunKernel([](sycl::queue q, sycl::range<1> range, auto reduction,
                           const auto& kernel) {
          syclex::submit(q, [&](sycl::handler& h) {
            syclex::parallel_for(h, range, kernel, reduction);
          });
        }));

  CHECK(RunKernel([](sycl::queue q, sycl::range<1> range, auto reduction,
                     const auto& kernel) {
          q.parallel_for(range, reduction, kernel);
        }) == RunKernel([](sycl::queue q, sycl::range<1> range, auto reduction,
                           const auto& kernel) {
          syclex::parallel_for(q, syclex::launch_config{range}, kernel,
                               reduction);
        }));

  CHECK(RunKernel([](sycl::queue q, sycl::range<1> range, auto reduction,
                     const auto& kernel) {
          q.submit([&](sycl::handler& h) {
            h.parallel_for(range, reduction, kernel);
          });
        }) == RunKernel([](sycl::queue q, sycl::range<1> range, auto reduction,
                           const auto& kernel) {
          syclex::submit(q, [&](sycl::handler& h) {
            syclex::parallel_for(h, syclex::launch_config{range}, kernel,
                                 reduction);
          });
        }));
}

static void test_nd_launch() {
  const auto Sum = [](const auto& enqueue, const std::vector<int>& input) {
    const int N = input.size();
    sycl::queue q;
    int* data = sycl::malloc_shared<int>(N, q);
    std::copy(input.begin(), input.end(), data);

    int* sum = sycl::malloc_shared<int>(1, q);
    auto reduction = sycl::reduction(sum, std::plus<>());
    sum[0] = 0;

    enqueue(q, sycl::nd_range<1>(N, 1), reduction,
            [=](sycl::nd_item<1> it, auto& sum) {
              sum.combine(data[it.get_global_id()]);
            });
    q.wait();

    const int output = sum[0];
    sycl::free(sum, q);
    return output;
  };

  const auto RunKernel = [&](const auto& enqueue) {
    const std::vector<int> input{1, 2, 3, 4};
    return Sum(enqueue, input);
  };

  CHECK(RunKernel([](sycl::queue q, sycl::nd_range<1> range, auto reduction,
                     const auto& kernel) {
          q.parallel_for(range, reduction, kernel);
        }) == RunKernel([](sycl::queue q, sycl::nd_range<1> range,
                           auto reduction, const auto& kernel) {
          syclex::nd_launch(q, range, kernel, reduction);
        }));

  CHECK(RunKernel([](sycl::queue q, sycl::nd_range<1> range, auto reduction,
                     const auto& kernel) {
          q.submit([&](sycl::handler& h) {
            h.parallel_for(range, reduction, kernel);
          });
        }) == RunKernel([](sycl::queue q, sycl::nd_range<1> range,
                           auto reduction, const auto& kernel) {
          syclex::submit(q, [&](sycl::handler& h) {
            syclex::nd_launch(h, range, kernel, reduction);
          });
        }));

  CHECK(RunKernel([](sycl::queue q, sycl::nd_range<1> range, auto reduction,
                     const auto& kernel) {
          q.parallel_for(range, reduction, kernel);
        }) == RunKernel([](sycl::queue q, sycl::nd_range<1> range,
                           auto reduction, const auto& kernel) {
          syclex::nd_launch(q, syclex::launch_config{range}, kernel, reduction);
        }));

  CHECK(RunKernel([](sycl::queue q, sycl::nd_range<1> range, auto reduction,
                     const auto& kernel) {
          q.submit([&](sycl::handler& h) {
            h.parallel_for(range, reduction, kernel);
          });
        }) == RunKernel([](sycl::queue q, sycl::nd_range<1> range,
                           auto reduction, const auto& kernel) {
          syclex::submit(q, [&](sycl::handler& h) {
            syclex::nd_launch(h, syclex::launch_config{range}, kernel,
                              reduction);
          });
        }));
}

static void test_memcpy() {
  const auto TestMemcpy = [](auto memcpy) {
    constexpr int N = 100;
    sycl::queue q;
    int* source = sycl::malloc_shared<int>(N, q);
    int* destination = sycl::malloc_shared<int>(N, q);
    std::iota(source, source + N, 0);
    std::fill(destination, destination + N, 0);

    memcpy(q, destination, source, N * sizeof(*source));
    q.wait();

    for (int i = 0; i < N; i++) CHECK(source[i] == destination[i]);
    sycl::free(source, q);
    sycl::free(destination, q);
  };

  TestMemcpy([](sycl::queue q, auto dest, auto src, size_t n) {
    syclex::memcpy(q, dest, src, n);
  });

  TestMemcpy([](sycl::queue q, auto dest, auto src, size_t n) {
    syclex::submit(q,
                   [&](sycl::handler& h) { syclex::memcpy(h, dest, src, n); });
  });
}

static void test_copy() {
  const auto TestCopy = [](auto copy) {
    constexpr int N = 100;
    sycl::queue q;
    int* source = sycl::malloc_shared<int>(N, q);
    int* destination = sycl::malloc_shared<int>(N, q);
    std::iota(source, source + N, 0);
    std::fill(destination, destination + N, 0);

    copy(q, destination, source, N);
    q.wait();

    for (int i = 0; i < N; i++) CHECK(source[i] == destination[i]);
    sycl::free(source, q);
    sycl::free(destination, q);
  };

  TestCopy([](sycl::queue q, auto dest, auto src, size_t count) {
    syclex::copy(q, dest, src, count);
  });

  TestCopy([](sycl::queue q, auto dest, auto src, size_t count) {
    syclex::submit(
        q, [&](sycl::handler& h) { syclex::copy(h, dest, src, count); });
  });
}

static void test_memset() {
  const auto TestMemset = [](auto memset) {
    constexpr int N = 100;
    constexpr unsigned char value = 0xFF;
    sycl::queue q;
    unsigned char* buffer = sycl::malloc_shared<unsigned char>(N, q);
    std::fill(buffer, buffer + N, 0);

    memset(q, buffer, value, N * sizeof(*buffer));
    q.wait();

    for (int i = 0; i < N; i++) CHECK(buffer[i] == value);
    sycl::free(buffer, q);
  };

  TestMemset([](sycl::queue q, auto ptr, auto value, size_t n) {
    syclex::memset(q, ptr, value, n);
  });

  TestMemset([](sycl::queue q, auto ptr, auto value, size_t n) {
    syclex::submit(q,
                   [&](sycl::handler& h) { syclex::memset(h, ptr, value, n); });
  });
}

static void test_fill() {
  const auto TestFill = [](auto fill) {
    constexpr int N = 100;
    constexpr int value = 17;
    sycl::queue q;
    int* buffer = sycl::malloc_shared<int>(N, q);
    std::fill(buffer, buffer + N, 0);

    fill(q, buffer, value, N);
    q.wait();

    for (int i = 0; i < N; i++) CHECK(buffer[i] == value);
    sycl::free(buffer, q);
  };

  TestFill([](sycl::queue q, auto ptr, auto value, size_t n) {
    syclex::fill(q, ptr, value, n);
  });

  TestFill([](sycl::queue q, auto ptr, auto value, size_t n) {
    syclex::submit(q,
                   [&](sycl::handler& h) { syclex::fill(h, ptr, value, n); });
  });
}

static void test_prefetch() {
  sycl::queue q;
  int* buffer = sycl::malloc_shared<int>(1, q);
  CHECK_NOTHROW(syclex::prefetch(q, buffer, sizeof(*buffer)));
  CHECK_NOTHROW(syclex::submit(q, [&](sycl::handler& h) {
    syclex::prefetch(h, buffer, sizeof(*buffer));
  }));
  sycl::free(buffer, q);
}

static void test_mem_advise() {
  sycl::queue q;
  int* buffer = sycl::malloc_shared<int>(1, q);
  CHECK_NOTHROW(syclex::mem_advise(q, buffer, sizeof(*buffer), 1));
  CHECK_NOTHROW(syclex::submit(q, [&](sycl::handler& h) {
    syclex::mem_advise(h, buffer, sizeof(*buffer), 1);
  }));
  sycl::free(buffer, q);
}

#ifdef SYCL_EXT_ONEAPI_ENQUEUE_BARRIER
static void test_barrier() {
  sycl::queue q;
  bool* task_done = sycl::malloc_shared<bool>(1, q);
  bool* test_passed = sycl::malloc_shared<bool>(1, q);
  *task_done = false;
  *test_passed = false;
  syclex::single_task(q, [=] {
    float sum = 0;
    for (int i = 0; i < 1000; i++) sum += sycl::sqrt(float(i));
    *task_done = (sum > 0);
  });
  syclex::barrier(q);
  syclex::single_task(q, [=] { *test_passed = *task_done; });
  q.wait();
  CHECK(*task_done);
  CHECK(*test_passed);
  sycl::free(task_done, q);
  sycl::free(test_passed, q);
}

static void test_partial_barrier() {
  sycl::queue q;
  bool* task_done = sycl::malloc_shared<bool>(1, q);
  bool* test_passed = sycl::malloc_shared<bool>(1, q);
  *task_done = false;
  *test_passed = false;
  const auto event = syclex::submit_with_event(q, [&](sycl::handler& h) {
    syclex::single_task(h, [=] {
      float sum = 0;
      for (int i = 0; i < 1000; i++) sum += sycl::sqrt(float(i));
      *task_done = (sum > 0);
    });
  });
  syclex::partial_barrier(q, {event});
  syclex::single_task(q, [=] { *test_passed = *task_done; });
  q.wait();
  CHECK(*task_done);
  CHECK(*test_passed);
  sycl::free(task_done, q);
  sycl::free(test_passed, q);
}
#endif  // SYCL_EXT_ONEAPI_ENQUEUE_BARRIER
#endif  // SYCL_EXT_ONEAPI_ENQUEUE_FUNCTIONS

TEST_CASE("Test case for \"Enqueue Functions\" extension",
          "[oneapi_enqueue_functions]") {
#ifndef SYCL_EXT_ONEAPI_ENQUEUE_FUNCTIONS
  SKIP("SYCL_EXT_ONEAPI_ENQUEUE_FUNCTIONS is not defined");
#else
  SECTION("Single tasks") { test_single_task(); }
  SECTION("Basic kernels") { test_parallel_for(); }
  SECTION("ND-range kernels") { test_nd_launch(); }
  SECTION("Memory operations") {
    test_memcpy();
    test_copy();
    test_memset();
    test_fill();
    test_prefetch();
    test_mem_advise();
  }
#ifdef SYCL_EXT_ONEAPI_ENQUEUE_BARRIER
  SECTION("Command barriers") {
    test_barrier();
    test_partial_barrier();
  }
#endif  // SYCL_EXT_ONEAPI_ENQUEUE_BARRIER
#endif  // SYCL_EXT_ONEAPI_ENQUEUE_FUNCTIONS
}

}  // namespace enqueue_functions::tests

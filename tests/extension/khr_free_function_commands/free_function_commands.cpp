/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2025 The Khronos Group Inc.
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

namespace free_function_commands::tests {

#ifdef SYCL_KHR_FREE_FUNCTION_COMMANDS

template <int Dims>
constexpr size_t get_N(size_t Size) {
  if constexpr (Dims == 1) return Size;
  if constexpr (Dims == 2) return Size * Size;
  if constexpr (Dims == 3) return Size * Size * Size;
}

static void test_submit() {
  sycl::queue q;
  constexpr int val = 314;

  auto test = [&](auto func) {
    int data = 0;
    {
      sycl::buffer<int, 1> buf{&data, 1};
      auto cgf = ([&](sycl::handler& h) {
        sycl::accessor acc{buf, h, sycl::write_only};
        h.single_task<>([=] { acc[0] = val; });
      });
      func(q, cgf);
    }
    CHECK(data == val);
  };

  test([&](sycl::queue q, auto cgf) { sycl::khr::submit(q, cgf); });
  test([&](sycl::queue q, auto cgf) { sycl::khr::submit_tracked(q, cgf); });
}

template <size_t Dims>
static void test_launch_impl() {
  constexpr size_t Size = 4;
  constexpr size_t N = get_N<Dims>(Size);
  sycl::queue q;
  sycl::range<Dims> r =
      sycl_cts::util::get_cts_object::range<Dims>::get(Size, Size, Size);

  int data[N] = {0};
  {
    sycl::buffer<int, 1> buf{data, N};
    q.submit([&](sycl::handler& h) {
      sycl::accessor acc{buf, h, sycl::write_only};
      sycl::khr::launch(h, r, [=](auto item) {
        auto lin_idx = item.get_linear_id();
        acc[lin_idx] = lin_idx * 2;
      });
    });
  }
  for (int i = 0; i < N; ++i) CHECK(data[i] == i * 2);

  if (q.get_device().has(sycl::aspect::usm_shared_allocations)) {
    int* buf = sycl::malloc_shared<int>(N, q);
    sycl::khr::launch(q, r, [=](auto item) {
      auto lin_idx = item.get_linear_id();
      buf[lin_idx] = lin_idx * 2;
    });
    q.wait();
    for (int i = 0; i < N; ++i) CHECK(buf[i] == i * 2);
    sycl::free(buf, q);
  }
}

static void test_launch() {
  test_launch_impl<1>();
  test_launch_impl<2>();
  test_launch_impl<3>();
}

template <int Dims>
static void test_launch_reduce_impl() {
  constexpr size_t Size = 4;
  constexpr size_t N = get_N<Dims>(Size);
  constexpr int expected_res = (N - 1) * N / 2;
  sycl::queue q;

  const auto task = [=](sycl::item<Dims> item, auto& sum) {
    sum += item.get_linear_id();
  };

  sycl::range<Dims> r =
      sycl_cts::util::get_cts_object::range<Dims>::get(Size, Size, Size);

  int sumResult = 0;
  {
    sycl::buffer<int> sumBuf{&sumResult, 1};
    q.submit([&](sycl::handler& h) {
      sycl::khr::launch_reduce(h, r, task,
                               sycl::reduction(sumBuf, h, sycl::plus<>()));
    });
  }
  CHECK(sumResult == expected_res);

  if (q.get_device().has(sycl::aspect::usm_shared_allocations)) {
    int* sumPtr = sycl::malloc_shared<int>(1, q);
    sumPtr[0] = 0;
    sycl::khr::launch_reduce(q, r, task,
                             sycl::reduction(sumPtr, sycl::plus<>()));
    q.wait();
    CHECK(sumPtr[0] == expected_res);
    sycl::free(sumPtr, q);
  }
}

static void test_launch_reduce() {
  test_launch_reduce_impl<1>();
  test_launch_reduce_impl<2>();
  test_launch_reduce_impl<3>();
}

template <int Dims>
static void test_launch_grouped_impl() {
  constexpr size_t Size = 4;
  constexpr size_t N = get_N<Dims>(Size);
  sycl::queue q;

  sycl::range<Dims> r_glob =
      sycl_cts::util::get_cts_object::range<Dims>::get(Size, Size, Size);
  sycl::range<Dims> r_loc = sycl_cts::util::get_cts_object::range<Dims>::get(
      Size / 2, Size / 2, Size / 2);

  int data[N] = {0};
  {
    sycl::buffer<int, 1> buf{data, N};
    q.submit([&](sycl::handler& h) {
      sycl::accessor acc(buf, h, sycl::write_only);
      sycl::khr::launch_grouped(h, r_glob, r_loc,
                                [=](sycl::nd_item<Dims> item) {
                                  auto idx = item.get_global_linear_id();
                                  acc[idx] = idx * 2;
                                });
    });
  }
  for (int i = 0; i < N; ++i) CHECK(data[i] == i * 2);

  if (q.get_device().has(sycl::aspect::usm_shared_allocations)) {
    int* buf = sycl::malloc_shared<int>(N, q);
    sycl::khr::launch_grouped(q, r_glob, r_loc, [=](sycl::nd_item<Dims> item) {
      auto idx = item.get_global_linear_id();
      buf[idx] = idx * 2;
    });
    q.wait();

    for (int i = 0; i < N; ++i) CHECK(buf[i] == i * 2);
    sycl::free(buf, q);
  }
}

static void test_launch_grouped() {
  test_launch_grouped_impl<1>();
  test_launch_grouped_impl<2>();
  test_launch_grouped_impl<3>();
}

template <int Dims>
static void test_launch_grouped_reduce_impl() {
  constexpr size_t Size = 4;
  constexpr size_t N = get_N<Dims>(Size);
  constexpr int expected_res = (N - 1) * N / 2;
  sycl::queue q;

  const auto task = [=](sycl::nd_item<Dims> item, auto& sum) {
    sum += item.get_global_linear_id();
  };

  sycl::range<Dims> r_glob =
      sycl_cts::util::get_cts_object::range<Dims>::get(Size, Size, Size);
  sycl::range<Dims> r_loc = sycl_cts::util::get_cts_object::range<Dims>::get(
      Size / 2, Size / 2, Size / 2);

  int sumResult = 0;
  {
    sycl::buffer<int> sumBuf{&sumResult, 1};
    q.submit([&](sycl::handler& h) {
      sycl::khr::launch_grouped_reduce(
          h, r_glob, r_loc, task, sycl::reduction(sumBuf, h, sycl::plus<>()));
    });
  }
  CHECK(sumResult == expected_res);

  if (q.get_device().has(sycl::aspect::usm_shared_allocations)) {
    int* sumPtr = sycl::malloc_shared<int>(1, q);
    sumPtr[0] = 0;
    sycl::khr::launch_grouped_reduce(q, r_glob, r_loc, task,
                                     sycl::reduction(sumPtr, sycl::plus<>()));
    q.wait();
    CHECK(sumPtr[0] == expected_res);
    sycl::free(sumPtr, q);
  }
}

static void test_launch_grouped_reduce() {
  test_launch_grouped_reduce_impl<1>();
  test_launch_grouped_reduce_impl<2>();
  test_launch_grouped_reduce_impl<3>();
}

static void test_launch_task() {
  sycl::queue q;
  constexpr int val = 314;

  int data = 0;
  {
    sycl::buffer<int, 1> buf{&data, 1};
    q.submit([&](sycl::handler& h) {
      sycl::accessor acc{buf, h, sycl::write_only};
      sycl::khr::launch_task(h, [=] { acc[0] = val; });
    });
  }
  CHECK(data == val);

  if (q.get_device().has(sycl::aspect::usm_shared_allocations)) {
    int* buf = sycl::malloc_shared<int>(1, q);
    sycl::khr::launch_task(q, [=] { buf[0] = val; });
    q.wait();
    CHECK(buf[0] == val);
    sycl::free(buf, q);
  }
}

static void test_memcpy() {
  sycl::queue q;
  if (q.get_device().has(sycl::aspect::usm_shared_allocations)) {
    constexpr size_t N = 8;
    int* src = sycl::malloc_shared<int>(N, q);
    int* dst = sycl::malloc_shared<int>(N, q);
    std::iota(src, src + N, 0);

    {
      q.submit([&](sycl::handler& h) {
        sycl::khr::memcpy(h, dst, src, N * sizeof(*src));
      });
      q.wait();

      for (int i = 0; i < N; ++i) {
        CHECK(src[i] == dst[i]);
        dst[i] = 0;
      }
    }
    {
      sycl::khr::memcpy(q, dst, src, N * sizeof(*src));
      q.wait();

      for (int i = 0; i < N; ++i) {
        CHECK(src[i] == dst[i]);
        dst[i] = 0;
      }
    }

    sycl::free(src, q);
    sycl::free(dst, q);
  }
}

template <typename T>
static void test_copy_usm_pointers_impl() {
  sycl::queue q;
  if (q.get_device().has(sycl::aspect::usm_shared_allocations)) {
    constexpr size_t N = 8;
    T* src = sycl::malloc_shared<T>(N, q);
    T* dst = sycl::malloc_shared<T>(N, q);
    std::iota(src, src + N, 0);

    {
      q.submit([&](sycl::handler& h) { sycl::khr::copy(h, dst, src, N); });
      q.wait();
      for (int i = 0; i < N; ++i) {
        CHECK(src[i] == dst[i]);
        dst[i] = 0;
      }
    }
    {
      sycl::khr::copy(q, dst, src, N);
      q.wait();
      for (int i = 0; i < N; ++i) CHECK(src[i] == dst[i]);
    }

    sycl::free(src, q);
    sycl::free(dst, q);
  }
}

static void test_copy_usm_pointers() {
  test_copy_usm_pointers_impl<char>();
  test_copy_usm_pointers_impl<int>();
  test_copy_usm_pointers_impl<float>();
}

template <typename T>
static void test_copy_accessors_host_to_device_impl() {
  using accT = sycl::accessor<T, 1, sycl::access::mode::write,
                              sycl::access::target::device>;
  const size_t N = 8;
  sycl::queue q;

  const auto test = [&](const auto& src, bool use_handler) {
    T dst[N] = {0};
    {
      sycl::buffer<T, 1> buf(dst, sycl::range<1>(N));

      if (use_handler) {
        q.submit([&](sycl::handler& h) {
          accT acc(buf, h, sycl::range<1>(N));
          sycl::khr::copy(h, src, acc);
        });
      } else {
        accT acc(buf);
        sycl::khr::copy(q, src, acc);
      }
    }

    for (size_t i = 0; i < N; ++i) CHECK(src[i] == dst[i]);
  };

  T src[N] = {0};
  std::iota(&src[0], &src[0] + N, 0);

  std::shared_ptr<T[]> src_sptr(new T[N]());
  std::iota(src_sptr.get(), src_sptr.get() + N, 0);

  test(src, true);
  test(src_sptr, true);
  test(src, false);
  test(src_sptr, false);
}

static void test_copy_accessors_host_to_device() {
  test_copy_accessors_host_to_device_impl<char>();
  test_copy_accessors_host_to_device_impl<int>();
  test_copy_accessors_host_to_device_impl<float>();
}

template <typename T>
static void test_copy_accessors_device_to_host_impl() {
  using accT = sycl::accessor<T, 1, sycl::access::mode::read,
                              sycl::access::target::device>;
  const size_t N = 8;
  sycl::queue q;

  const auto test = [&](auto& dst, bool use_handler) {
    T src[N] = {0};
    std::iota(&src[0], &src[0] + N, 0);
    {
      sycl::buffer<T, 1> buf(src, sycl::range<1>(N));

      if (use_handler) {
        q.submit([&](sycl::handler& h) {
          accT acc(buf, h, sycl::range<1>(N));
          sycl::khr::copy(h, acc, dst);
        });
      } else {
        accT acc(buf);
        sycl::khr::copy(q, acc, dst);
      }
    }

    for (size_t i = 0; i < N; ++i) CHECK(src[i] == dst[i]);
  };

  T dst[N] = {0};
  std::shared_ptr<T[]> dst_sptr(new T[N]());

  test(dst, true);
  test(dst_sptr, true);
  test(dst, false);
  test(dst_sptr, false);
}

static void test_copy_accessors_device_to_host() {
  test_copy_accessors_device_to_host_impl<char>();
  test_copy_accessors_device_to_host_impl<int>();
  test_copy_accessors_device_to_host_impl<float>();
}

template <typename T>
static void test_copy_accessors_device_to_device_impl() {
  using acc_src_T = sycl::accessor<T, 1, sycl::access::mode::read,
                                   sycl::access::target::device>;
  using acc_dst_T = sycl::accessor<T, 1, sycl::access::mode::write,
                                   sycl::access::target::device>;
  const size_t N = 8;
  T src[N] = {0};
  std::iota(&src[0], &src[0] + N, 0);

  sycl::queue q;
  auto test_copy = [&](bool use_handler) {
    T dst[N] = {0};
    {
      sycl::buffer<T, 1> buf_src(src, sycl::range<1>(N));
      sycl::buffer<T, 1> buf_dst(dst, sycl::range<1>(N));

      if (use_handler) {
        q.submit([&](sycl::handler& h) {
          acc_src_T acc_src(buf_src, h, sycl::range<1>(N));
          acc_dst_T acc_dst(buf_dst, h, sycl::range<1>(N));
          sycl::khr::copy(h, acc_src, acc_dst);
        });
      } else {
        acc_src_T acc_src(buf_src, sycl::range<1>(N));
        acc_dst_T acc_dst(buf_dst, sycl::range<1>(N));
        sycl::khr::copy(q, acc_src, acc_dst);
      }
    }

    for (size_t i = 0; i < N; ++i) {
      CHECK(src[i] == dst[i]);
    }
  };

  test_copy(true);
  test_copy(false);
}

static void test_copy_accessors_device_to_device() {
  test_copy_accessors_device_to_device_impl<char>();
  test_copy_accessors_device_to_device_impl<int>();
  test_copy_accessors_device_to_device_impl<float>();
}

static void test_memset() {
  constexpr size_t N = 8;
  constexpr int val = 7;
  sycl::queue q;

  auto test_memset = [&](bool use_handler) {
    auto ptr = (char*)malloc_shared(N, q);
    if (use_handler)
      q.submit([&](sycl::handler& h) { sycl::khr::memset(h, ptr, val, N); });
    else
      sycl::khr::memset(q, ptr, val, N);
    q.wait();

    for (int i = 0; i < N; ++i) CHECK(ptr[i] == val);

    sycl::free(ptr, q);
  };

  test_memset(true);
  test_memset(false);
}

template <typename T>
static void test_fill_impl() {
  using accT = sycl::accessor<int, 1, sycl::access::mode::write,
                              sycl::access::target::device>;
  constexpr size_t N = 8;
  constexpr int val = 7;
  sycl::queue q;

  auto test_fill_shared = [&](bool use_handler) {
    auto ptr = sycl::malloc_shared<int>(N, q);
    if (use_handler)
      q.submit([&](sycl::handler& h) { sycl::khr::fill(h, ptr, val, N); });
    else
      sycl::khr::fill(q, ptr, val, N);
    q.wait();

    for (int i = 0; i < N; ++i) CHECK(ptr[i] == val);

    sycl::free(ptr, q);
  };

  auto test_fill_buffer = [&](bool use_handler) {
    int dst[N] = {0};
    {
      sycl::buffer<int, 1> buf(dst, sycl::range<1>(N));
      if (use_handler) {
        q.submit([&](sycl::handler& h) {
          accT acc(buf, h, sycl::range<1>(N));
          sycl::khr::fill(h, acc, val);
        });
      } else {
        accT acc(buf, sycl::range<1>(N));
        sycl::khr::fill(q, acc, val);
      }
    }

    for (int i = 0; i < N; ++i) CHECK(dst[i] == val);
  };

  if (q.get_device().has(sycl::aspect::usm_shared_allocations)) {
    test_fill_shared(true);
    test_fill_shared(false);
  }
  test_fill_buffer(true);
  test_fill_buffer(false);
}

static void test_fill() {
  test_fill_impl<char>();
  test_fill_impl<int>();
  test_fill_impl<float>();
}

template <typename T>
static void test_update_host_impl() {
  const size_t N = 8;
  using accT = sycl::accessor<T, 1, sycl::access::mode::write,
                              sycl::access::target::device>;
  sycl::queue q;

  auto test_buffer = [&](bool use_handler) {
    T data[N] = {0};
    {
      sycl::buffer<T, 1> buf(data, sycl::range<1>(N));

      q.submit([&](sycl::handler& h) {
        accT acc(buf, h, sycl::range<1>(N));
        h.parallel_for(sycl::range<1>{N},
                       [=](sycl::id<1> idx) { acc[idx] = idx; });
      });

      if (use_handler) {
        q.submit([&](sycl::handler& h) {
          accT acc(buf, h, sycl::range<1>(N));
          sycl::khr::update_host(h, acc);
        });
      } else {
        accT acc(buf, sycl::range<1>(N));
        sycl::khr::update_host(q, acc);
      }
    }

    for (size_t i = 0; i < N; ++i) CHECK(data[i] == i);
  };

  test_buffer(true);
  test_buffer(false);
}

static void test_update_host() {
  test_update_host_impl<char>();
  test_update_host_impl<int>();
  test_update_host_impl<float>();
}

static void test_prefetch() {
  sycl::queue q;
  if (q.get_device().has(sycl::aspect::usm_shared_allocations)) {
    int* buffer = sycl::malloc_shared<int>(1, q);
    CHECK_NOTHROW(sycl::khr::prefetch(q, buffer, sizeof(*buffer)));
    CHECK_NOTHROW(q.submit([&](sycl::handler& h) {
      sycl::khr::prefetch(h, buffer, sizeof(*buffer));
    }));
    sycl::free(buffer, q);
  }
}

static void test_mem_advise() {
  sycl::queue q;
  if (q.get_device().has(sycl::aspect::usm_shared_allocations)) {
    int* buffer = sycl::malloc_shared<int>(1, q);
    CHECK_NOTHROW(sycl::khr::mem_advise(q, buffer, sizeof(*buffer), 1));
    CHECK_NOTHROW(q.submit([&](sycl::handler& h) {
      sycl::khr::mem_advise(h, buffer, sizeof(*buffer), 1);
    }));
    sycl::free(buffer, q);
  }
}

static void test_command_barrier() {
  sycl::queue q;
  if (q.get_device().has(sycl::aspect::usm_shared_allocations)) {
    bool* task_done = sycl::malloc_shared<bool>(1, q);
    bool* test_passed = sycl::malloc_shared<bool>(1, q);
    *task_done = false;
    *test_passed = false;

    q.single_task([=] {
      float sum = 0;
      for (int i = 0; i < 10000; ++i) sum += sycl::sqrt(float(i));
      *task_done = (sum > 0);
    });

    sycl::khr::command_barrier(q);

    q.single_task([=] { *test_passed = *task_done; });
    q.wait();

    CHECK(*task_done);
    CHECK(*test_passed);
    sycl::free(task_done, q);
    sycl::free(test_passed, q);
  }
}

static void test_event_barrier() {
  sycl::queue q;
  if (q.get_device().has(sycl::aspect::usm_shared_allocations)) {
    bool* task_done = sycl::malloc_shared<bool>(1, q);
    bool* test_passed = sycl::malloc_shared<bool>(1, q);
    *task_done = false;
    *test_passed = false;
    const auto event = q.submit([&](sycl::handler& h) {
      h.single_task([=] {
        float sum = 0;
        for (int i = 0; i < 10000; ++i) sum += sycl::sqrt(float(i));
        *task_done = (sum > 0);
      });
    });
    sycl::khr::event_barrier(q, {event});
    q.single_task([=] { *test_passed = *task_done; });
    q.wait();
    CHECK(*task_done);
    CHECK(*test_passed);

    sycl::free(task_done, q);
    sycl::free(test_passed, q);
  }
}
#endif  // SYCL_KHR_FREE_FUNCTION_COMMANDS

TEST_CASE("Test case for SYCL_KHR_FREE_FUNCTION_COMMANDS extension",
          "[SYCL_KHR_FREE_FUNCTION_COMMANDS]") {
#ifndef SYCL_KHR_FREE_FUNCTION_COMMANDS
  SKIP("SYCL_KHR_FREE_FUNCTION_COMMANDS is not defined");
#else
  SECTION("Command-groups") { test_submit(); }

  SECTION("Kernel launch") {
    test_launch();
    test_launch_reduce();
    test_launch_grouped();
    test_launch_grouped_reduce();
    test_launch_task();
  }

  SECTION("Memory operations") {
    test_memcpy();
    test_copy_usm_pointers();
    test_copy_accessors_host_to_device();
    test_copy_accessors_device_to_host();
    test_copy_accessors_device_to_device();
    test_memset();
    test_fill();
    test_update_host();
    test_prefetch();
    test_mem_advise();
  }

  SECTION("Memory operations") {
    test_command_barrier();
    test_event_barrier();
  }
#endif  // SYCL_KHR_FREE_FUNCTION_COMMANDS
}
}  // namespace free_function_commands::tests

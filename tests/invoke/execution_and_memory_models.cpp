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
#include "../common/disabled_for_test_case.h"

template <int n>
class kernel;

template <typename acc_t>
void long_loop(acc_t loop_acc, size_t buffer_size) {
  for (int i = 0; i < 1000000; i++) {
    int s = sycl::sqrt(float(i));
    loop_acc[s % buffer_size] = i;
  }
}

TEST_CASE("Execution order of three command groups submitted to the same queue",
          "[invoke]") {
  const size_t buffer_size = 12;
  auto queue = sycl_cts::util::get_cts_object::queue();

  std::array<int, buffer_size> data1;
  std::array<int, buffer_size> data2;
  std::array<int, buffer_size> res1;
  std::array<int, buffer_size> res2;

  int loop_array[buffer_size];
  {
    sycl::buffer<int, 1> buffer1(data1.data(), sycl::range(buffer_size));
    sycl::buffer<int, 1> buffer2(data2.data(), sycl::range(buffer_size));
    sycl::buffer<int, 1> res_buffer1(res1.data(), sycl::range(buffer_size));
    sycl::buffer<int, 1> res_buffer2(res2.data(), sycl::range(buffer_size));
    sycl::buffer<int, 1> loop_buf(loop_array, sycl::range(buffer_size));

    queue.submit([&](sycl::handler& cgh) {
      auto acc1 =
          buffer1.template get_access<sycl::access_mode::read_write>(cgh);
      auto loop_acc = loop_buf.get_access<sycl::access_mode::read_write>(cgh);
      cgh.single_task<kernel<1>>([=] {
        long_loop(loop_acc, buffer_size);
        for (int i = buffer_size - 1; i >= 0; i--) acc1[i] = i;
      });
    });
    queue.submit([&](sycl::handler& cgh) {
      auto acc2 =
          buffer2.template get_access<sycl::access_mode::read_write>(cgh);
      auto loop_acc = loop_buf.get_access<sycl::access_mode::read_write>(cgh);
      cgh.single_task<kernel<2>>([=] {
        long_loop(loop_acc, buffer_size);
        for (int i = buffer_size - 1; i >= 0; i--) acc2[i] = i;
      });
    });
    queue.submit([&](sycl::handler& cgh) {
      auto acc1 =
          buffer1.template get_access<sycl::access_mode::read_write>(cgh);
      auto acc2 =
          buffer2.template get_access<sycl::access_mode::read_write>(cgh);
      auto res_acc1 =
          res_buffer1.template get_access<sycl::access_mode::read_write>(cgh);
      auto res_acc2 =
          res_buffer2.template get_access<sycl::access_mode::read_write>(cgh);
      cgh.parallel_for<kernel<3>>(sycl::range<1>(buffer_size), [=](sycl::id<1> index) {
        res_acc1[index] = acc1[index];
        res_acc2[index] = acc2[index];
      });
    });
    queue.wait_and_throw();
  }

  CHECK(data1 == res1);
  CHECK(data2 == res2);
}

TEST_CASE(
    "Execution order of three command groups submitted to the different queues",
    "[invoke]") {
  const std::vector<sycl::device> devices{sycl::device::get_devices()};

  if (devices.size() >= 3) {
    const size_t buffer_size = 12;
    sycl::queue q0(devices[0]);
    sycl::queue q1(devices[1]);
    sycl::queue q2(devices[2]);

    std::array<int, buffer_size> data1;
    std::array<int, buffer_size> data2;
    std::array<int, buffer_size> res1;
    std::array<int, buffer_size> res2;

    int loop_array[buffer_size];
    {
      sycl::buffer<int, 1> buffer1(data1.data(), sycl::range(buffer_size));
      sycl::buffer<int, 1> buffer2(data2.data(), sycl::range(buffer_size));
      sycl::buffer<int, 1> res_buffer1(res1.data(), sycl::range(buffer_size));
      sycl::buffer<int, 1> res_buffer2(res2.data(), sycl::range(buffer_size));
      sycl::buffer<int, 1> loop_buf(loop_array, sycl::range(buffer_size));

      q0.submit([&](sycl::handler& cgh) {
        auto acc1 =
            buffer1.template get_access<sycl::access_mode::read_write>(cgh);
        auto loop_acc = loop_buf.get_access<sycl::access_mode::read_write>(cgh);
        cgh.single_task<kernel<4>>([=] {
          long_loop(loop_acc, buffer_size);
          for (int i = buffer_size - 1; i >= 0; i--) acc1[i] = i;
        });
      });
      q1.submit([&](sycl::handler& cgh) {
        auto acc2 =
            buffer2.template get_access<sycl::access_mode::read_write>(cgh);
        auto loop_acc = loop_buf.get_access<sycl::access_mode::read_write>(cgh);
        cgh.single_task<kernel<5>>([=] {
          long_loop(loop_acc, buffer_size);
          for (int i = buffer_size - 1; i >= 0; i--) acc2[i] = i;
        });
      });
      q2.submit([&](sycl::handler& cgh) {
        auto acc1 =
            buffer1.template get_access<sycl::access_mode::read_write>(cgh);
        auto acc2 =
            buffer2.template get_access<sycl::access_mode::read_write>(cgh);
        auto res_acc1 =
            res_buffer1.template get_access<sycl::access_mode::read_write>(cgh);
        auto res_acc2 =
            res_buffer2.template get_access<sycl::access_mode::read_write>(cgh);
        cgh.parallel_for<kernel<6>>(sycl::range<1>(buffer_size), [=](sycl::id<1> index) {
          res_acc1[index] = acc1[index];
          res_acc2[index] = acc2[index];
        });
      });
    }

    CHECK(data1 == res1);
    CHECK(data2 == res2);

  } else {
    SKIP("Test requires 3 or more devices");
  }
}

DISABLED_FOR_TEST_CASE(hipSYCL)
("Requirements on overlapping sub-buffers", "[invoke]") ({
  auto device = sycl_cts::util::get_cts_object::device();
  auto queue = sycl_cts::util::get_cts_object::queue();

  if (device.has(sycl::aspect::usm_device_allocations)) {
    const auto mem_base_addr_align =
        device.get_info<sycl::info::device::mem_base_addr_align>();
    const size_t buffer_size = mem_base_addr_align * 3;

    bool res = false;
    int data[buffer_size];
    int loop_array[buffer_size];
    {
      sycl::buffer<bool, 1> res_buffer(&res, sycl::range<1>(1));
      sycl::buffer<int, 1> buffer(data, sycl::range<1>(buffer_size));
      sycl::buffer<int, 1> sub_buf1(buffer, 0, mem_base_addr_align * 2);
      sycl::buffer<int, 1> sub_buf2(buffer, mem_base_addr_align,
                                    mem_base_addr_align * 2);
      sycl::buffer<int, 1> loop_buf(loop_array, sycl::range(buffer_size));

      int* pflag = sycl::malloc_device<int>(1, queue);
      // assign pflag value to 0
      queue
          .submit(
              [&](sycl::handler& cgh) { cgh.single_task<kernel<7>>([=] { *pflag = 0; }); })
          .wait();
      queue.submit([&](sycl::handler& cgh) {
        auto acc1 =
            sub_buf1.template get_access<sycl::access_mode::read_write>(cgh);
        auto loop_acc = loop_buf.get_access<sycl::access_mode::read_write>(cgh);
        cgh.single_task<kernel<8>>([=] {
          sycl::atomic_ref<int, sycl::memory_order::relaxed,
                           sycl::memory_scope::device>
              rflag(*pflag);
          for (int i = acc1.size() - 1; i >= 0; i--) acc1[i] = i;
          long_loop(loop_acc, buffer_size);
          rflag = 1;
        });
      });
      queue.submit([&](sycl::handler& cgh) {
        auto acc2 =
            sub_buf2.template get_access<sycl::access_mode::read_write>(cgh);
        auto res_acc =
            res_buffer.template get_access<sycl::access_mode::read_write>(cgh);
        cgh.single_task<kernel<9>>([=] {
          sycl::atomic_ref<int, sycl::memory_order::relaxed,
                           sycl::memory_scope::device>
              rflag(*pflag);
          res_acc[0] = (rflag == 1);
          for (int i = acc2.size() - 1; i >= 0; i--) acc2[i] = i;
        });
      });
      queue.wait_and_throw();
    }

    CHECK(res);

  } else {
    SKIP("Device does not support USM device allocations");
  }
});

TEST_CASE("Host accessor as a barrier", "[invoke]") {
  auto device = sycl_cts::util::get_cts_object::device();
  auto queue = sycl_cts::util::get_cts_object::queue();

  if (device.has(sycl::aspect::usm_atomic_shared_allocations)) {
    const size_t buffer_size = 12;
    int data[buffer_size];
    int loop_array[buffer_size];

    {
      sycl::buffer<int, 1> buffer(data, sycl::range<1>(buffer_size));
      sycl::buffer<int, 1> loop_buf(loop_array, sycl::range<1>(buffer_size));
      int* pflag = sycl::malloc_shared<int>(1, queue);

      queue.submit([&](sycl::handler& cgh) {
        auto acc =
            buffer.template get_access<sycl::access_mode::read_write>(cgh);
        auto loop_acc =
            loop_buf.template get_access<sycl::access_mode::read_write>(cgh);
        cgh.single_task<kernel<10>>([=] {
          long_loop(loop_acc, buffer_size);
          for (int i = buffer_size - 1; i >= 0; i--) acc[i] = i;
          *pflag = 42;
        });
      });
      sycl::host_accessor<int, 1> host_acc(buffer);
      CHECK(*pflag == 42);
    }
  } else {
    SKIP("Device does not support USM atomic shared allocations");
  }
}

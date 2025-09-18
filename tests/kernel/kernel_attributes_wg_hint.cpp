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
#include "kernel_attributes.h"
#include <cmath>

using namespace kernel_attributes;

static constexpr int size = 4;
static auto test_max_wg_size = pow(size * 2, 3);

static bool device_supports_test_max_wg_size() {
  static bool result =
      sycl_cts::util::get_cts_object::device()
          .get_info<sycl::info::device::max_work_group_size>() >=
      test_max_wg_size;
  return result;
}

template <int Dimensions, int Size>
class functor {
 public:
  static constexpr int dims = Dimensions;
  static constexpr int size = Size;

  using acc_t =
      sycl::accessor<int, dims, sycl::access_mode::write, sycl::target::device>;

  functor(acc_t _acc) : acc(_acc) {}

  [[sycl::work_group_size_hint(size)]] void operator()(
      sycl::nd_item<1> nd_item) const {
    acc[nd_item.get_local_id()] = expected_val<1>();
  }

  [[sycl::work_group_size_hint(size, size)]] void operator()(
      sycl::nd_item<2> nd_item) const {
    acc[nd_item.get_local_id()] = expected_val<2>();
  }

  [[sycl::work_group_size_hint(size, size, size)]] void operator()(
      sycl::nd_item<3> nd_item) const {
    acc[nd_item.get_local_id()] = expected_val<3>();
  }

  [[sycl::work_group_size_hint(size)]] void operator()(
      sycl::group<1> group) const {
    acc[group.get_group_id()] = expected_val<1>();
  }

  [[sycl::work_group_size_hint(size, size)]] void operator()(
      sycl::group<2> group) const {
    acc[group.get_group_id()] = expected_val<2>();
  }

  [[sycl::work_group_size_hint(size, size, size)]] void operator()(
      sycl::group<3> group) const {
    acc[group.get_group_id()] = expected_val<3>();
  }

 private:
  acc_t acc;
};

template <int dims, int size, class acc_t>
const auto get_lambda(acc_t& acc) {
  if constexpr (dims == 1) {
    return [=](auto nd_item) [[sycl::work_group_size_hint(size)]] {
      acc[nd_item.get_local_id()] = expected_val<dims>();
    };
  } else if constexpr (dims == 2) {
    return [=](auto nd_item) [[sycl::work_group_size_hint(size, size)]] {
      acc[nd_item.get_local_id()] = expected_val<dims>();
    };
  } else if constexpr (dims == 3) {
    return [=](auto nd_item) [[sycl::work_group_size_hint(size, size, size)]] {
      acc[nd_item.get_local_id()] = expected_val<dims>();
    };
  } else {
    return [=](auto nd_item) {};
  }
}

template <int dims, int size, class acc_t>
const auto get_lambda_wg(acc_t& acc) {
  if constexpr (dims == 1) {
    return [=](auto group) [[sycl::work_group_size_hint(size)]] {
      acc[group.get_group_id()] = expected_val<dims>();
    };
  } else if constexpr (dims == 2) {
    return [=](auto group) [[sycl::work_group_size_hint(size, size)]] {
      acc[group.get_group_id()] = expected_val<dims>();
    };
  } else if constexpr (dims == 3) {
    return [=](auto group) [[sycl::work_group_size_hint(size, size, size)]] {
      acc[group.get_group_id()] = expected_val<dims>();
    };
  } else {
    return [=](auto group) {};
  }
}

template <int dims, class arr_t>
void verify(arr_t arr, const char* msg) {
  INFO(
      "Check that kernel is executed without any "
      "exception and has expected result using "
      << msg);
  CHECK(std::all_of(arr.cbegin(), arr.cend(),
                    [](const int i) { return i == expected_val<dims>(); }));
}

template <class functor>
void check_functor_and_sep_lambda() {
  auto queue = sycl_cts::util::get_cts_object::queue();

  constexpr int dims = functor::dims;
  constexpr int size = functor::size;
  constexpr int buffer_size = (dims == 3) ? size * size * size
                              : dims == 2 ? size * size
                                          : size;

  std::array<int, buffer_size> data_functor;
  std::array<int, buffer_size> data_sep_lambda;

  std::array<int, buffer_size> data_functor_wg;
  std::array<int, buffer_size> data_sep_lambda_wg;

  const auto range =
      sycl_cts::util::get_cts_object::range<dims>::get(size, size, size);
  const auto range_wg =
      sycl_cts::util::get_cts_object::range<dims>::get(1, 1, 1);
  {
    sycl::buffer<int, dims> buf1(data_functor.data(), range);
    sycl::buffer<int, dims> buf2(data_sep_lambda.data(), range);

    sycl::buffer<int, dims> buf1_wg(data_functor_wg.data(), range);
    sycl::buffer<int, dims> buf2_wg(data_sep_lambda_wg.data(), range);

    // functor
    queue.submit([&](sycl::handler& cgh) {
      auto acc1 = sycl::accessor(buf1, cgh, sycl::write_only);
      cgh.parallel_for<kernel_functor<functor>>(sycl::nd_range{range, range},
                                                functor{acc1});
    });
    queue.submit([&](sycl::handler& cgh) {
      auto acc1_wg = sycl::accessor(buf1_wg, cgh, sycl::write_only);
      cgh.parallel_for_work_group<kernel_functor_wg<functor>>(range, range_wg,
                                                              functor{acc1_wg});
    });

    // separate lambda
    queue.submit([&](sycl::handler& cgh) {
      auto acc2 = sycl::accessor(buf2, cgh, sycl::write_only);
      cgh.parallel_for<kernel_separate_lambda<dims, size>>(
          sycl::nd_range{range, range}, get_lambda<dims, size>(acc2));
    });
    queue.submit([&](sycl::handler& cgh) {
      auto acc2_wg = sycl::accessor(buf2_wg, cgh, sycl::write_only);
      cgh.parallel_for_work_group<kernel_separate_lambda_wg<dims, size>>(
          range, range_wg, get_lambda_wg<dims, size>(acc2_wg));
    });
    queue.wait_and_throw();
  }

  verify<dims>(data_functor, "functor with nd_item");
  verify<dims>(data_functor_wg, "functor with group");
  verify<dims>(data_sep_lambda, "separate lambda with nd_item");
  verify<dims>(data_sep_lambda_wg, "separate lambda with group");
}

template <int size>
void run_tests_for_lambda() {
  auto queue = sycl_cts::util::get_cts_object::queue();

  constexpr int buffer_size_1d = size;
  constexpr int buffer_size_2d = size * size;
  constexpr int buffer_size_3d = size * size * size;

  std::array<int, buffer_size_1d> data_1d;
  std::array<int, buffer_size_2d> data_2d;
  std::array<int, buffer_size_3d> data_3d;

  std::array<int, buffer_size_1d> data_wg_1d;
  std::array<int, buffer_size_2d> data_wg_2d;
  std::array<int, buffer_size_3d> data_wg_3d;

  sycl::range<1> range_1d(size);
  sycl::range<2> range_2d(size, size);
  sycl::range<3> range_3d(size, size, size);
  {
    sycl::buffer<int, 1> buf_1d(data_1d.data(), range_1d);
    sycl::buffer<int, 2> buf_2d(data_2d.data(), range_2d);
    sycl::buffer<int, 3> buf_3d(data_3d.data(), range_3d);

    sycl::buffer<int, 1> buf_wg_1d(data_wg_1d.data(), range_1d);
    sycl::buffer<int, 2> buf_wg_2d(data_wg_2d.data(), range_2d);
    sycl::buffer<int, 3> buf_wg_3d(data_wg_3d.data(), range_3d);

    // lambda submission call
    queue.submit([&](sycl::handler& cgh) {
      auto acc_1d = sycl::accessor(buf_1d, cgh, sycl::write_only);
      cgh.parallel_for<kernel_lambda<1, buffer_size_1d>>(
          sycl::nd_range<1>{range_1d, range_1d},
          [=](auto nd_item) [[sycl::work_group_size_hint(size)]] {
            acc_1d[nd_item.get_local_id()] = expected_val<1>();
          });
    });
    queue.submit([&](sycl::handler& cgh) {
      auto acc_2d = sycl::accessor(buf_2d, cgh, sycl::write_only);
      cgh.parallel_for<kernel_lambda<2, buffer_size_2d>>(
          sycl::nd_range<2>{range_2d, range_2d},
          [=](auto nd_item) [[sycl::work_group_size_hint(size, size)]] {
            acc_2d[nd_item.get_local_id()] = expected_val<2>();
          });
    });
    queue.submit([&](sycl::handler& cgh) {
      auto acc_wg_1d = sycl::accessor(buf_wg_1d, cgh, sycl::write_only);
      cgh.parallel_for_work_group<kernel_lambda_wg<1, buffer_size_1d>>(
          range_1d, sycl::range<1>{1},
          [=](auto group) [[sycl::work_group_size_hint(size)]] {
            acc_wg_1d[group.get_group_id()] = expected_val<1>();
          });
    });
    queue.submit([&](sycl::handler& cgh) {
      auto acc_wg_2d = sycl::accessor(buf_wg_2d, cgh, sycl::write_only);
      cgh.parallel_for_work_group<kernel_lambda_wg<2, buffer_size_2d>>(
          range_2d, sycl::range<2>{1, 1},
          [=](auto group) [[sycl::work_group_size_hint(size, size)]] {
            acc_wg_2d[group.get_group_id()] = expected_val<2>();
          });
    });

    if (device_supports_test_max_wg_size()) {
      queue.submit([&](sycl::handler& cgh) {
        auto acc_3d = sycl::accessor(buf_3d, cgh, sycl::write_only);
        cgh.parallel_for<kernel_lambda<3, buffer_size_3d>>(
            sycl::nd_range<3>{range_3d, range_3d},
            [=](auto nd_item) [[sycl::work_group_size_hint(size, size, size)]] {
              acc_3d[nd_item.get_local_id()] = expected_val<3>();
            });
      });
      queue.submit([&](sycl::handler& cgh) {
        auto acc_wg_3d = sycl::accessor(buf_wg_3d, cgh, sycl::write_only);
        cgh.parallel_for_work_group<kernel_lambda_wg<3, buffer_size_3d>>(
            range_3d, sycl::range<3>{1, 1, 1},
            [=](auto group) [[sycl::work_group_size_hint(size, size, size)]] {
              acc_wg_3d[group.get_group_id()] = expected_val<3>();
            });
      });
    } else {
      WARN("Device does not support work group size " << test_max_wg_size);
    }
    queue.wait_and_throw();
  }

  verify<1>(data_1d, "lambda with nd_item 1 dim");
  verify<2>(data_2d, "lambda with nd_item 2 dims");

  verify<1>(data_wg_1d, "lambda with group 1 dim");
  verify<2>(data_wg_2d, "lambda with group 2 dims");

  if (device_supports_test_max_wg_size()) {
    verify<3>(data_wg_3d, "lambda with group 3 dims");
    verify<3>(data_3d, "lambda with nd_item 3 dims");
  }
}

template <int dims>
void run_tests_for_dim() {
  check_functor_and_sep_lambda<functor<dims, size>>();
  check_functor_and_sep_lambda<functor<dims, size / 2>>();

  if (device_supports_test_max_wg_size()) {
    check_functor_and_sep_lambda<functor<dims, size * 2>>();
  } else {
    WARN("Device does not support work group size " << test_max_wg_size);
  }
}

TEST_CASE("Behavior of kernel attribute work_group_size_hint", "[kernel]") {
  run_tests_for_dim<1>();
  run_tests_for_dim<2>();
  run_tests_for_dim<3>();

  run_tests_for_lambda<size>();
  run_tests_for_lambda<size / 2>();
  run_tests_for_lambda<size * 2>();
}

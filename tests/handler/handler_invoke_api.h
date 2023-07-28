/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2022-2023 The Khronos Group Inc.
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

constexpr size_t bufferSize = 256;
constexpr size_t defaultNumModified = 128;

constexpr sycl::specialization_id<int> SpecName(5);

struct TestConstants {
  const sycl::range<1> defaultRange;
  const sycl::id<1> offset;
  const sycl::range<1> offsetRange;

  const sycl::nd_range<1> ndRange;
  const sycl::nd_range<1> offsetNdRange;
  const sycl::range<1> numWorkGroups;
  const sycl::range<1> workGroupSize;
  TestConstants()
      : defaultRange(defaultNumModified),
        offset(28),
        offsetRange(defaultRange[0] - offset[0]),
        ndRange(defaultRange, defaultRange),
        offsetNdRange(offsetRange, offsetRange, offset),
        numWorkGroups(1),
        workGroupSize(defaultRange) {}
};

struct use_offset {
  static const int no = 0;
  static const int yes = 1;
};

class CustomNdItem {
  sycl::nd_item<1> item;

 public:
  CustomNdItem(const sycl::nd_item<1>& rt_item) : item(rt_item) {}
  sycl::id<1> get_global_id() { return item.get_global_id(); }
  size_t get_global_id(int dim) { return item.get_global_id(dim); }
};

class kernel_test_class0;
class kernel_test_class1;
class kernel_test_class2;
class kernel_test_class3;
class kernel_test_class4;
class kernel_test_class5;

using accessor_t =
    sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::target::device>;

struct single_task_functor {
  single_task_functor(accessor_t acc, size_t range) : acc(acc), range(range) {}

  void operator()() const {
    for (size_t i = 0; i < range; ++i) {
      acc[i] = i;
    }
  }

  accessor_t acc;
  size_t range;
};

template <int useOffset>
struct parallel_for_range_id_functor {
  parallel_for_range_id_functor(accessor_t acc) : acc(acc) {}

  void operator()(sycl::id<1> id) const { acc[id] = id[0]; }

  accessor_t acc;
};

template <int useOffset>
struct parallel_for_range_item_functor {
  parallel_for_range_item_functor(accessor_t acc) : acc(acc) {}

  void operator()(sycl::item<1> item) const { acc[item] = item[0]; }

  accessor_t acc;
};

template <int useOffset>
struct parallel_for_range_auto_functor {
  parallel_for_range_auto_functor(accessor_t acc) : acc(acc) {}

  template <typename ItemT>
  void operator()(ItemT item) const {
    acc[item] = item[0];
  }

  accessor_t acc;
};

template <int useOffset>
struct parallel_for_range_size_t_functor {
  parallel_for_range_size_t_functor(accessor_t acc) : acc(acc) {}

  void operator()(size_t ind) const { acc[ind] = ind; }

  accessor_t acc;
};

template <int useOffset>
struct parallel_for_range_item_functor_with_kernel_handler {
  parallel_for_range_item_functor_with_kernel_handler(accessor_t acc)
      : acc(acc) {}

  void operator()(sycl::item<1> item, sycl::kernel_handler kh) const {
    kh.get_specialization_constant<SpecName>();
    acc[item] = item[0];
  }

  accessor_t acc;
};

struct parallel_for_nd_range_nd_item_functor {
  parallel_for_nd_range_nd_item_functor(accessor_t acc) : acc(acc) {}

  void operator()(sycl::nd_item<1> ndItem) const {
    acc[ndItem.get_global_id()] = ndItem.get_global_id(0);
  }

  accessor_t acc;
};

struct parallel_for_nd_range_auto_functor {
  parallel_for_nd_range_auto_functor(accessor_t acc) : acc(acc) {}

  template <typename ItemT>
  void operator()(ItemT ndItem) const {
    acc[ndItem.get_global_id()] = ndItem.get_global_id(0);
  }

  accessor_t acc;
};

struct parallel_for_nd_range_custom_nd_item_functor {
  parallel_for_nd_range_custom_nd_item_functor(accessor_t acc) : acc(acc) {}

  void operator()(CustomNdItem ndItem) const {
    acc[ndItem.get_global_id()] = ndItem.get_global_id(0);
  }

  accessor_t acc;
};

struct parallel_for_nd_range_nd_item_functor_with_kernel_handler {
  parallel_for_nd_range_nd_item_functor_with_kernel_handler(accessor_t acc)
      : acc(acc) {}

  void operator()(sycl::nd_item<1> ndItem, sycl::kernel_handler kh) const {
    kh.get_specialization_constant<SpecName>();
    acc[ndItem.get_global_id()] = ndItem.get_global_id(0);
  }

  accessor_t acc;
};

#define PARALLEL_FOR_WORK_GROUP_DYNAMIC_FUNCTOR_BODY                          \
  group.parallel_for_work_item([&](sycl::h_item<1> item) { acc[0] = 0; });    \
  group.parallel_for_work_item(sycl::range<1>{1}, [&](sycl::h_item<1> item) { \
    if (item.get_global_id(0) > 0) {                                          \
      acc[item.get_global_id()] = item.get_global_id(0);                      \
    }                                                                         \
  })

/**
 * Functor for testing calls to parallel_for_work_group with a
 * dynamic (runtime-defined) work group size.
 */
struct parallel_for_work_group_dynamic_functor {
  parallel_for_work_group_dynamic_functor(accessor_t acc) : acc(acc) {}

  void operator()(sycl::group<1> group) const {
    PARALLEL_FOR_WORK_GROUP_DYNAMIC_FUNCTOR_BODY;
  }

  accessor_t acc;
};

/**
 * Functor for testing calls to parallel_for_work_group with a
 * dynamic (runtime-defined) work group size and with kernel handler.
 */
struct parallel_for_work_group_dynamic_with_kern_handler_functor {
  parallel_for_work_group_dynamic_with_kern_handler_functor(accessor_t acc)
      : acc(acc) {}

  void operator()(sycl::group<1> group, sycl::kernel_handler kh) const {
    kh.get_specialization_constant<SpecName>();
    PARALLEL_FOR_WORK_GROUP_DYNAMIC_FUNCTOR_BODY;
  }

  accessor_t acc;
};

#define PARALLEL_FOR_WORK_GROUP_FIXED_FUNCTOR_BODY                            \
  group.parallel_for_work_item([&](sycl::h_item<1> item) {                    \
    if (item.get_global_id(0) > 0) {                                          \
      acc[item.get_global_id()] = item.get_global_id(0);                      \
    }                                                                         \
  });                                                                         \
  group.parallel_for_work_item(sycl::range<1>{1}, [&](sycl::h_item<1> item) { \
    acc[item.get_global_id()] = 0;                                            \
  })

/**
 * Functor for testing calls to parallel_for_work_group with a
 * fixed work group size.
 */
struct parallel_for_work_group_fixed_functor {
  explicit parallel_for_work_group_fixed_functor(accessor_t acc) : acc(acc) {}

  void operator()(sycl::group<1> group) const {
    PARALLEL_FOR_WORK_GROUP_FIXED_FUNCTOR_BODY;
  }

  accessor_t acc{};
};

/**
 * Functor for testing calls to parallel_for_work_group with a
 * fixed work group size and with kernel handler.
 */
struct parallel_for_work_group_fixed_with_kern_handler_functor {
  explicit parallel_for_work_group_fixed_with_kern_handler_functor(
      accessor_t acc)
      : acc(acc) {}

  void operator()(sycl::group<1> group, sycl::kernel_handler kh) const {
    kh.get_specialization_constant<SpecName>();
    PARALLEL_FOR_WORK_GROUP_FIXED_FUNCTOR_BODY;
  }

  accessor_t acc{};
};

/**
 * @brief Checks that the handler API call works correctly
 * @tparam kernel_wrapper Type of the lambda used to invoke the API method
 * @param methodName Name of the method to be checked
 * @param queue Queue to submit the kernel to
 * @param kernelWrapper A lambda that contains a call to a handler API method.
 * @param startIndex The first index that contains a modified value.
 * @param numModified The number of elements that contain modified values.
 *
 * The lambda is passed a handler and a read_write accessor. The elements
 * inside the range [startIndex, startIndex + numModified] must be assigned
 * their corresponding global work item id.
 */
template <class kernel_wrapper>
void check_api_call(std::string methodName, sycl::queue& queue,
                    kernel_wrapper&& kernelWrapper, size_t startIndex = 0,
                    size_t numModified = defaultNumModified) {
  INFO("Check " << methodName);
  // Initialize buffer with a canary value we can recognize again below.
  std::vector<int> result(bufferSize, 12345);
  {
    auto buf = sycl::buffer<int, 1>(result.data(), result.size());
    queue.submit([&](sycl::handler& cgh) {
      auto acc = buf.get_access<sycl::access_mode::read_write>(cgh);
      kernelWrapper(cgh, acc);
    });
  }
  for (size_t i = 0; i < bufferSize; ++i) {
    INFO("For element " << i);
    if (i < startIndex || i >= startIndex + numModified) {
      CHECK((12345 == result[i]));
    } else {
      CHECK((i == result[i]));
    }
  }
}

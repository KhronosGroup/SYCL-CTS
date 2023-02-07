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

struct use_offset {
  static const int no = 0;
  static const int yes = 1;
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

struct parallel_for_nd_range_functor {
  parallel_for_nd_range_functor(accessor_t acc) : acc(acc) {}

  void operator()(sycl::nd_item<1> ndItem) const {
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

class single_task_kernel;
class parallel_for_range_id_kernel;
class parallel_for_range_offset_id_kernel;
class parallel_for_range_item_kernel;
class parallel_for_range_offset_item_kernel;
class parallel_for_nd_range_kernel;
class parallel_for_nd_range_offset_kernel;
class parallel_for_work_group_1range_kernel;
class parallel_for_work_group_2range_kernel;

static constexpr size_t bufferSize = 256;
static constexpr size_t defaultNumModified = 128;

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
void check_api_call(std::string methodName, sycl::queue &queue,
                    kernel_wrapper &&kernelWrapper, size_t startIndex = 0,
                    size_t numModified = defaultNumModified) {
  INFO("Check " << methodName);
  // Initialize buffer with a canary value we can recognize again below.
  std::vector<int> result(bufferSize, 12345);
  {
    auto buf = sycl::buffer<int, 1>(result.data(), result.size());
    queue.submit([&](sycl::handler &cgh) {
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

TEST_CASE("handler_invoke_api", "[handler]") {
  using handler = sycl::handler;

  auto queue = sycl_cts::util::get_cts_object::queue();
  auto deviceList = queue.get_context().get_devices();

  const sycl::range<1> defaultRange = defaultNumModified;
  const sycl::id<1> offset = 28;
  const sycl::range<1> offsetRange = defaultRange[0] - offset[0];

  const sycl::nd_range<1> ndRange(defaultRange, defaultRange);
  const sycl::nd_range<1> offsetNdRange(offsetRange, offsetRange, offset);
  const sycl::range<1> numWorkGroups(1);
  const sycl::range<1> workGroupSize(defaultRange);

  /* single_task */
  check_api_call("single_task(lambda)", queue,
                 [&](handler &cgh, accessor_t acc) {
                   cgh.single_task<class single_task_kernel>([=]() {
                     single_task_functor f(acc, defaultRange[0]);
                     f();
                   });
                 });
  check_api_call("single_task(functor)", queue,
                 [&](handler &cgh, accessor_t acc) {
                   cgh.single_task<single_task_functor>(
                       single_task_functor(acc, defaultRange[0]));
                 });
#if SYCL_CTS_ENABLE_FEATURE_SET_FULL
  check_api_call("single_task(lambda), no kernel name", queue,
                 [&](handler &cgh, accessor_t acc) {
                   cgh.single_task([=]() {
                     single_task_functor f(acc, defaultRange[0]);
                     f();
                   });
                 });
  check_api_call("single_task(functor), no kernel name", queue,
                 [&](handler &cgh, accessor_t acc) {
                   cgh.single_task(single_task_functor(acc, defaultRange[0]));
                 });
#endif

  /* parallel_for with id */
  check_api_call("parallel_for(range, lambda) with id", queue,
                 [&](handler &cgh, accessor_t acc) {
                   cgh.parallel_for<class parallel_for_range_id_kernel>(
                       defaultRange, [=](sycl::id<1> id) {
                         parallel_for_range_id_functor<use_offset::no> f(acc);
                         f(id);
                       });
                 });
  check_api_call("parallel_for(range, functor) with id", queue,
                 [&](handler &cgh, accessor_t acc) {
                   using functor =
                       parallel_for_range_id_functor<use_offset::no>;
                   cgh.parallel_for<functor>(defaultRange, functor(acc));
                 });
#if SYCL_CTS_ENABLE_FEATURE_SET_FULL
  check_api_call("parallel_for(range, lambda) with id, no kernel name", queue,
                 [&](handler &cgh, accessor_t acc) {
                   cgh.parallel_for(defaultRange, [=](sycl::id<1> id) {
                     parallel_for_range_id_functor<use_offset::no> f(acc);
                     f(id);
                   });
                 });
  check_api_call("parallel_for(range, functor) with id, no kernel name", queue,
                 [&](handler &cgh, accessor_t acc) {
                   cgh.parallel_for(
                       defaultRange,
                       parallel_for_range_id_functor<use_offset::no>(acc));
                 });
#endif

  /* parallel_for with offset and id */
#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
  check_api_call(
      "parallel_for(range, id, lambda) with id", queue,
      [&](handler &cgh, accessor_t acc) {
        cgh.parallel_for<class parallel_for_range_offset_id_kernel>(
            offsetRange, offset, [=](sycl::id<1> id) {
              parallel_for_range_id_functor<use_offset::yes> f(acc);
              f(id);
            });
      },
      offset[0], offsetRange[0]);
  check_api_call(
      "parallel_for(range, id, functor) with id", queue,
      [&](handler &cgh, accessor_t acc) {
        using functor = parallel_for_range_id_functor<use_offset::yes>;
        cgh.parallel_for<functor>(offsetRange, offset, functor(acc));
      },
      offset[0], offsetRange[0]);
#if SYCL_CTS_ENABLE_FEATURE_SET_FULL
  check_api_call(
      "parallel_for(range, id, lambda) with id, no kernel name", queue,
      [&](handler &cgh, accessor_t acc) {
        cgh.parallel_for(offsetRange, offset, [=](sycl::id<1> id) {
          parallel_for_range_id_functor<use_offset::yes> f(acc);
          f(id);
        });
      },
      offset[0], offsetRange[0]);
  check_api_call(
      "parallel_for(range, id, functor) with id, no kernel name", queue,
      [&](handler &cgh, accessor_t acc) {
        cgh.parallel_for(offsetRange, offset,
                         parallel_for_range_id_functor<use_offset::yes>(acc));
      },
      offset[0], offsetRange[0]);
#endif  // SYCL_CTS_ENABLE_FEATURE_SET_FULL
#endif  // SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS

  /* parallel_for with item */
  check_api_call("parallel_for(range, lambda) with item", queue,
                 [&](handler &cgh, accessor_t acc) {
                   cgh.parallel_for<class parallel_for_range_item_kernel>(
                       defaultRange, [=](sycl::item<1> item) {
                         parallel_for_range_item_functor<use_offset::no> f(acc);
                         f(item);
                       });
                 });
  check_api_call("parallel_for(range, functor) with item", queue,
                 [&](handler &cgh, accessor_t acc) {
                   using functor =
                       parallel_for_range_item_functor<use_offset::no>;
                   cgh.parallel_for<functor>(defaultRange, functor(acc));
                 });
#if SYCL_CTS_ENABLE_FEATURE_SET_FULL
  check_api_call("parallel_for(range, lambda) with item, no kernel name", queue,
                 [&](handler &cgh, accessor_t acc) {
                   cgh.parallel_for(defaultRange, [=](sycl::item<1> item) {
                     parallel_for_range_item_functor<use_offset::no> f(acc);
                     f(item);
                   });
                 });
  check_api_call("parallel_for(range, functor) with item, no kernel name",
                 queue, [&](handler &cgh, accessor_t acc) {
                   cgh.parallel_for(
                       defaultRange,
                       parallel_for_range_item_functor<use_offset::no>(acc));
                 });
#endif

  /* parallel_for with offset and item */
#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
  check_api_call(
      "parallel_for(range, id, lambda) with item", queue,
      [&](handler &cgh, accessor_t acc) {
        cgh.parallel_for<class parallel_for_range_offset_item_kernel>(
            offsetRange, offset, [=](sycl::item<1> item) {
              parallel_for_range_item_functor<use_offset::yes> f(acc);
              f(item);
            });
      },
      offset[0], offsetRange[0]);
  check_api_call(
      "parallel_for(range, id, functor) with item", queue,
      [&](handler &cgh, accessor_t acc) {
        using functor = parallel_for_range_item_functor<use_offset::yes>;
        cgh.parallel_for<functor>(offsetRange, offset, functor(acc));
      },
      offset[0], offsetRange[0]);
#if SYCL_CTS_ENABLE_FEATURE_SET_FULL
  check_api_call(
      "parallel_for(range, id, lambda) with item, no kernel name", queue,
      [&](handler &cgh, accessor_t acc) {
        cgh.parallel_for(offsetRange, offset, [=](sycl::item<1> item) {
          parallel_for_range_item_functor<use_offset::yes> f(acc);
          f(item);
        });
      },
      offset[0], offsetRange[0]);
  check_api_call(
      "parallel_for(range, id, functor) with item, no kernel name", queue,
      [&](handler &cgh, accessor_t acc) {
        cgh.parallel_for(offsetRange, offset,
                         parallel_for_range_item_functor<use_offset::yes>(acc));
      },
      offset[0], offsetRange[0]);
#endif  // SYCL_CTS_ENABLE_FEATURE_SET_FULL
#endif  // SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS

  /* parallel_for over nd_range without offset */
  check_api_call("parallel_for(nd_range, lambda)", queue,
                 [&](handler &cgh, accessor_t acc) {
                   cgh.parallel_for<class parallel_for_nd_range_kernel>(
                       ndRange, [=](sycl::nd_item<1> ndItem) {
                         parallel_for_nd_range_functor f(acc);
                         f(ndItem);
                       });
                 });
  check_api_call("parallel_for(nd_range, functor)", queue,
                 [&](handler &cgh, accessor_t acc) {
                   cgh.parallel_for<parallel_for_nd_range_functor>(
                       ndRange, parallel_for_nd_range_functor(acc));
                 });
#if SYCL_CTS_ENABLE_FEATURE_SET_FULL
  check_api_call("parallel_for(nd_range, lambda), no kernel name", queue,
                 [&](handler &cgh, accessor_t acc) {
                   cgh.parallel_for(ndRange, [=](sycl::nd_item<1> ndItem) {
                     parallel_for_nd_range_functor f(acc);
                     f(ndItem);
                   });
                 });
  check_api_call("parallel_for(nd_range, functor), no kernel name", queue,
                 [&](handler &cgh, accessor_t acc) {
                   cgh.parallel_for(ndRange,
                                    parallel_for_nd_range_functor(acc));
                 });
#endif

  /* parallel_for over nd_range with offset */
#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
  check_api_call(
      "parallel_for(nd_range, lambda) with offset", queue,
      [&](handler &cgh, accessor_t acc) {
        cgh.parallel_for<class parallel_for_nd_range_offset_kernel>(
            offsetNdRange, [=](sycl::nd_item<1> ndItem) {
              parallel_for_nd_range_functor f(acc);
              f(ndItem);
            });
      },
      offset[0], offsetRange[0]);
  check_api_call(
      "parallel_for(nd_range, functor) with offset", queue,
      [&](handler &cgh, accessor_t acc) {
        cgh.parallel_for<parallel_for_nd_range_functor>(
            offsetNdRange, parallel_for_nd_range_functor(acc));
      },
      offset[0], offsetRange[0]);
#if SYCL_CTS_ENABLE_FEATURE_SET_FULL
  check_api_call(
      "parallel_for(nd_range, lambda) with offset, no kernel name", queue,
      [&](handler &cgh, accessor_t acc) {
        cgh.parallel_for(offsetNdRange, [=](sycl::nd_item<1> ndItem) {
          parallel_for_nd_range_functor f(acc);
          f(ndItem);
        });
      },
      offset[0], offsetRange[0]);
  check_api_call(
      "parallel_for(nd_range, functor) with offset, no kernel name", queue,
      [&](handler &cgh, accessor_t acc) {
        cgh.parallel_for(offsetNdRange, parallel_for_nd_range_functor(acc));
      },
      offset[0], offsetRange[0]);
#endif  // SYCL_CTS_ENABLE_FEATURE_SET_FULL
#endif  // SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS

  /* parallel_for_work_group (range) */
  check_api_call("parallel_for_work_group(range, lambda)", queue,
                 [&](handler &cgh, accessor_t acc) {
                   cgh.parallel_for_work_group<
                       class parallel_for_work_group_1range_kernel>(
                       defaultRange, [=](sycl::group<1> group) {
                         // cannot instantiate functor as parallel_for_work_item
                         // must be invoked from parallel_for_work_group context
                         PARALLEL_FOR_WORK_GROUP_DYNAMIC_FUNCTOR_BODY;
                       });
                 });
  check_api_call(
      "parallel_for_work_group(range, functor)", queue,
      [&](handler &cgh, accessor_t acc) {
        cgh.parallel_for_work_group<parallel_for_work_group_dynamic_functor>(
            defaultRange, parallel_for_work_group_dynamic_functor(acc));
      });
#if SYCL_CTS_ENABLE_FEATURE_SET_FULL
  check_api_call("parallel_for_work_group(range, lambda), no kernel name",
                 queue, [&](handler &cgh, accessor_t acc) {
                   cgh.parallel_for_work_group(
                       defaultRange, [=](sycl::group<1> group) {
                         // cannot instantiate functor as parallel_for_work_item
                         // must be invoked from parallel_for_work_group context
                         PARALLEL_FOR_WORK_GROUP_DYNAMIC_FUNCTOR_BODY;
                       });
                 });
  check_api_call("parallel_for_work_group(range, functor), no kernel name",
                 queue, [&](handler &cgh, accessor_t acc) {
                   cgh.parallel_for_work_group(
                       defaultRange,
                       parallel_for_work_group_dynamic_functor(acc));
                 });
#endif

  /* parallel_for_work_group (range, range) */
  check_api_call("parallel_for_work_group(range, range, lambda)", queue,
                 [&](handler &cgh, accessor_t acc) {
                   cgh.parallel_for_work_group<
                       class parallel_for_work_group_2range_kernel>(
                       numWorkGroups, workGroupSize, [=](sycl::group<1> group) {
                         // cannot instantiate functor as parallel_for_work_item
                         // must be invoked from parallel_for_work_group context
                         PARALLEL_FOR_WORK_GROUP_FIXED_FUNCTOR_BODY;
                       });
                 });
  check_api_call(
      "parallel_for_work_group(range, range, functor)", queue,
      [&](handler &cgh, accessor_t acc) {
        cgh.parallel_for_work_group<parallel_for_work_group_fixed_functor>(
            numWorkGroups, workGroupSize,
            parallel_for_work_group_fixed_functor(acc));
      });
#if SYCL_CTS_ENABLE_FEATURE_SET_FULL
  check_api_call(
      "parallel_for_work_group(range, range, lambda), no kernel name", queue,
      [&](handler &cgh, accessor_t acc) {
        cgh.parallel_for_work_group(
            numWorkGroups, workGroupSize, [=](sycl::group<1> group) {
              // cannot instantiate functor as parallel_for_work_item
              // must be invoked from parallel_for_work_group context
              PARALLEL_FOR_WORK_GROUP_FIXED_FUNCTOR_BODY;
            });
      });
  check_api_call(
      "parallel_for_work_group(range, range, functor), no kernel name", queue,
      [&](handler &cgh, accessor_t acc) {
        cgh.parallel_for_work_group(numWorkGroups, workGroupSize,
                                    parallel_for_work_group_fixed_functor(acc));
      });
#endif

  /* single_task with kernel object */
  if (!is_compiler_available(deviceList)) {
    WARN(
        "online compiler is not available -- skipping test of "
        "single_task with kernel object");
  } else {
    {
      using k_name = kernel_test_class0;
      check_api_call("single_task<kernel_test_class>()", queue,
                     [&](handler &cgh, accessor_t acc) {
                       cgh.single_task<k_name>([=]() {
                         for (size_t i = 0; i < defaultRange[0]; ++i) {
                           acc[i] = i;
                         }
                       });
                     });
    }
  }

  /* parallel_for with kernel object */
  if (!is_compiler_available(deviceList)) {
    WARN(
        "online compiler is not available -- skipping test of "
        "parallel_for with kernel object");
  } else {
    {
      using k_name = kernel_test_class1;
      check_api_call("parallel_for(range, kernel) with id", queue,
                     [&](handler &cgh, accessor_t acc) {
                       cgh.parallel_for<k_name>(
                           defaultRange,
                           [=](sycl::id<1> id) { acc[id] = id[0]; });
                     });
    }

#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
    {
      using k_name = kernel_test_class2;
      check_api_call(
          "parallel_for(range, offset, kernel) with id", queue,
          [&](handler &cgh, accessor_t acc) {
            cgh.parallel_for<k_name>(offsetRange, offset,
                                     [=](sycl::id<1> id) { acc[id] = id[0]; });
          },
          offset[0], offsetRange[0]);
    }
#endif

    {
      using k_name = kernel_test_class3;
      check_api_call("parallel_for(range, kernel) with item", queue,
                     [&](handler &cgh, accessor_t acc) {
                       cgh.parallel_for<k_name>(
                           defaultRange,
                           [=](sycl::item<1> item) { acc[item] = item[0]; });
                     });
    }

#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
    {
      using k_name = kernel_test_class4;
      check_api_call(
          "parallel_for(range, offset, kernel) with item", queue,
          [&](handler &cgh, accessor_t acc) {
            cgh.parallel_for<k_name>(
                offsetRange, offset,
                [=](sycl::item<1> item) { acc[item] = item[0]; });
          },
          offset[0], offsetRange[0]);
    }
#endif

    {
      using k_name = kernel_test_class5;
      check_api_call(
          "parallel_for(nd_range, kernel);", queue,
          [&](handler &cgh, accessor_t acc) {
            cgh.parallel_for<k_name>(ndRange, [=](sycl::nd_item<1> ndItem) {
              acc[ndItem.get_global_id()] = ndItem.get_global_id(0);
            });
          });
    }
  }
}

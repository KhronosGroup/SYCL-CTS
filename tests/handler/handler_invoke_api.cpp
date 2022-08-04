/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
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

#include "../common/common.h"

#define TEST_NAME handler_invoke_api

struct use_offset {
  static const int no = 0;
  static const int yes = 1;
};

namespace handler_invoke_api__ {
using namespace sycl_cts;

class kernel_test_class0;
class kernel_test_class1;
class kernel_test_class2;
class kernel_test_class3;
class kernel_test_class4;
class kernel_test_class5;

using accessor_t =
    cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                       cl::sycl::access::target::global_buffer>;

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

  void operator()(cl::sycl::id<1> id) const { acc[id] = id[0]; }

  accessor_t acc;
};

template <int useOffset>
struct parallel_for_range_item_functor {
  parallel_for_range_item_functor(accessor_t acc) : acc(acc) {}

  void operator()(cl::sycl::item<1> item) const { acc[item] = item[0]; }

  accessor_t acc;
};

struct parallel_for_nd_range_functor {
  parallel_for_nd_range_functor(accessor_t acc) : acc(acc) {}

  void operator()(cl::sycl::nd_item<1> ndItem) const {
    acc[ndItem.get_global_id()] = ndItem.get_global_id(0);
  }

  accessor_t acc;
};

/**
 * Functor for testing calls to parallel_for_work_group with a
 * dynamic (runtime-defined) work group size.
 */
struct parallel_for_work_group_dynamic_functor {
  parallel_for_work_group_dynamic_functor(accessor_t acc) : acc(acc) {}

  void operator()(cl::sycl::group<1> group) const {
    group.parallel_for_work_item([&](cl::sycl::h_item<1> item) { acc[0] = 0; });

    cl::sycl::range<1> subRange(1);

    group.parallel_for_work_item(subRange, [&](cl::sycl::h_item<1> item) {
      if (item.get_global_id(0) > 0) {
        acc[item.get_global_id()] = item.get_global_id(0);
      }
    });
  }

  accessor_t acc;
};

/**
 * Functor for testing calls to parallel_for_work_group with a
 * fixed work group size.
 */
struct parallel_for_work_group_fixed_functor {
  parallel_for_work_group_fixed_functor(accessor_t acc) : acc(acc) {}

  void operator()(cl::sycl::group<1> group) const {
    group.parallel_for_work_item([&](cl::sycl::h_item<1> item) {
      if (item.get_global_id(0) > 0) {
        acc[item.get_global_id()] = item.get_global_id(0);
      }
    });

    cl::sycl::range<1> subRange(1);

    group.parallel_for_work_item(subRange, [&](cl::sycl::h_item<1> item) {
      acc[item.get_global_id()] = 0;
    });
  }

  accessor_t acc;
};

class single_task_lambda_prebuilt;
template <int useOffset>
class parallel_for_range_id_lambda_prebuilt;
template <int useOffset>
class parallel_for_range_item_lambda_prebuilt;
class parallel_for_nd_range_lambda_prebuilt;
class parallel_for_nd_range_offset_lambda_prebuilt;
class parallel_for_work_group_1range_lambda_prebuilt;
class parallel_for_work_group_2range_lambda_prebuilt;

/**
 * @brief Retrieves a prebuilt kernel
 * @tparam kernel_name Name of the kernel to retrieve
 * @param queue Queue that contains the context for the kernel
 * @return The built kernel
 */
template <class kernel_name>
cl::sycl::kernel get_prebuilt_kernel(cl::sycl::queue &queue) {
  return util::get_cts_object::kernel::prebuilt<kernel_name>(queue);
}

class single_task_lambda;
class parallel_for_range_id_lambda;
class parallel_for_range_offset_id_lambda;
class parallel_for_range_item_lambda;
class parallel_for_range_offset_item_lambda;
class parallel_for_nd_range_lambda;
class parallel_for_nd_range_offset_lambda;
class parallel_for_work_group_1range_lambda;
class parallel_for_work_group_2range_lambda;

/** tests the invoke APIs
 */
class TEST_NAME : public sycl_cts::util::test_base {
 public:
  static constexpr size_t bufferSize = 256;
  static constexpr size_t defaultNumModified = 128;

  /**
   * @brief Checks that the handler API call works correctly
   * @tparam kernel_wrapper Type of the lambda used to invoke the API method
   * @param methodName Name of the method to be checked
   * @param log Test logger object
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
  void check_api_call(cl::sycl::string_class methodName, util::logger &log,
                      cl::sycl::queue &queue, kernel_wrapper &&kernelWrapper,
                      size_t startIndex = 0,
                      size_t numModified = defaultNumModified) {
    log.note("Check %s", methodName.c_str());
    // Initialize buffer with a canary value we can recognize again below.
    std::vector<int> result(bufferSize, 12345);
    {
      auto buf = cl::sycl::buffer<int, 1>(result.data(), result.size());
      queue.submit([&](cl::sycl::handler &cgh) {
        auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
        kernelWrapper(cgh, acc);
      });
    }
    for (size_t i = 0; i < bufferSize; ++i) {
      if (i < startIndex || i >= startIndex + numModified) {
        if (!CHECK_VALUE(log, result[i], 12345, i)) return;
      } else {
        if (!CHECK_VALUE(log, result[i], i, i)) return;
      }
    }
  }

  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute this test
   */
  void run(util::logger &log) override {
    try {
      using handler = cl::sycl::handler;

      auto queue = util::get_cts_object::queue();
      auto deviceList = queue.get_context().get_devices();

      const cl::sycl::range<1> defaultRange = defaultNumModified;
      const cl::sycl::id<1> offset = 28;
      const cl::sycl::range<1> offsetRange = defaultRange[0] - offset[0];

      const cl::sycl::nd_range<1> ndRange(defaultRange, defaultRange);
      const cl::sycl::nd_range<1> offsetNdRange(offsetRange, offsetRange,
                                                offset);
      const cl::sycl::range<1> numWorkGroups(1);
      const cl::sycl::range<1> workGroupSize(defaultRange);

      /* single_task */

      check_api_call("single_task(lambda)", log, queue,
                     [&](handler &cgh, accessor_t acc) {
                       cgh.single_task<class single_task_lambda>([=]() {
                         for (size_t i = 0; i < defaultRange[0]; ++i) {
                           acc[i] = i;
                         }
                       });
                     });

      check_api_call(
          "single_task(functor)", log, queue,
          [&](handler &cgh, accessor_t acc) {
            cgh.single_task(single_task_functor(acc, defaultRange[0]));
          });

      if (!is_compiler_available(deviceList)) {
        log.note(
            "online compiler is not available -- skipping "
            "test of single_task with prebuilt kernel");
      } else {
        {
          auto preBuiltKernel =
              get_prebuilt_kernel<single_task_lambda_prebuilt>(queue);

          check_api_call("single_task(kernel, lambda)", log, queue,
                         [&](handler &cgh, accessor_t acc) {
                           cgh.single_task<single_task_lambda_prebuilt>(
                               preBuiltKernel, [=]() {
                                 for (size_t i = 0; i < defaultRange[0]; ++i) {
                                   acc[i] = i;
                                 }
                               });
                         });
        }

        {
          auto preBuiltKernel = get_prebuilt_kernel<single_task_functor>(queue);

          check_api_call("single_task(kernel, functor)", log, queue,
                         [&](handler &cgh, accessor_t acc) {
                           cgh.single_task<single_task_functor>(
                               preBuiltKernel,
                               single_task_functor(acc, defaultRange[0]));
                         });
        }
      }

      /* parallel_for with id */

      check_api_call("parallel_for(range, lambda) with id", log, queue,
                     [&](handler &cgh, accessor_t acc) {
                       cgh.parallel_for<class parallel_for_range_id_lambda>(
                           defaultRange,
                           [=](cl::sycl::id<1> id) { acc[id] = id[0]; });
                     });

      check_api_call("parallel_for(range, functor) with id", log, queue,
                     [&](handler &cgh, accessor_t acc) {
                       cgh.parallel_for(
                           defaultRange,
                           parallel_for_range_id_functor<use_offset::no>(acc));
                     });

      if (!is_compiler_available(deviceList)) {
        log.note(
            "online compiler is not available -- skipping "
            "test of parallel_for with id with prebuilt kernel");
      } else {
        {
          auto preBuiltKernel = get_prebuilt_kernel<
              parallel_for_range_id_lambda_prebuilt<use_offset::no>>(queue);

          check_api_call(
              "parallel_for(kernel, range, lambda) with id", log, queue,
              [&](handler &cgh, accessor_t acc) {
                cgh.parallel_for<
                    parallel_for_range_id_lambda_prebuilt<use_offset::no>>(
                    preBuiltKernel, defaultRange,
                    [=](cl::sycl::id<1> id) { acc[id] = id[0]; });
              });
        }

        {
          auto preBuiltKernel = get_prebuilt_kernel<
              parallel_for_range_id_functor<use_offset::no>>(queue);

          check_api_call(
              "parallel_for(kernel, range, functor) with id", log, queue,
              [&](handler &cgh, accessor_t acc) {
                cgh.parallel_for(
                    preBuiltKernel, defaultRange,
                    parallel_for_range_id_functor<use_offset::no>(acc));
              });
        }
      }

      /* parallel_for with offset and id */

      check_api_call(
          "parallel_for(range, id, lambda) with id", log, queue,
          [&](handler &cgh, accessor_t acc) {
            cgh.parallel_for<class parallel_for_range_offset_id_lambda>(
                offsetRange, offset,
                [=](cl::sycl::id<1> id) { acc[id] = id[0]; });
          },
          offset[0], offsetRange[0]);

      check_api_call(
          "parallel_for(range, id, functor) with id", log, queue,
          [&](handler &cgh, accessor_t acc) {
            cgh.parallel_for(
                offsetRange, offset,
                parallel_for_range_id_functor<use_offset::yes>(acc));
          },
          offset[0], offsetRange[0]);

      if (!is_compiler_available(deviceList)) {
        log.note(
            "online compiler is not available -- skipping "
            "test of parallel_for with offset and id with "
            "prebuilt kernel");
      } else {
        {
          auto preBuiltKernel = get_prebuilt_kernel<
              parallel_for_range_id_lambda_prebuilt<use_offset::yes>>(queue);

          check_api_call(
              "parallel_for(kernel, range, id, lambda) with id", log, queue,
              [&](handler &cgh, accessor_t acc) {
                cgh.parallel_for<
                    parallel_for_range_id_lambda_prebuilt<use_offset::yes>>(
                    preBuiltKernel, offsetRange, offset,
                    [=](cl::sycl::id<1> id) { acc[id] = id[0]; });
              },
              offset[0], offsetRange[0]);
        }

        {
          auto preBuiltKernel = get_prebuilt_kernel<
              parallel_for_range_id_functor<use_offset::yes>>(queue);

          check_api_call(
              "parallel_for(kernel, range, id, functor) with id", log, queue,
              [&](handler &cgh, accessor_t acc) {
                cgh.parallel_for(
                    preBuiltKernel, offsetRange, offset,
                    parallel_for_range_id_functor<use_offset::yes>(acc));
              },
              offset[0], offsetRange[0]);
        }
      }

      /* parallel_for with item */

      check_api_call("parallel_for(range, lambda) with item", log, queue,
                     [&](handler &cgh, accessor_t acc) {
                       cgh.parallel_for<class parallel_for_range_item_lambda>(
                           defaultRange, [=](cl::sycl::item<1> item) {
                             acc[item] = item[0];
                           });
                     });

      check_api_call(
          "parallel_for(range, functor) with item", log, queue,
          [&](handler &cgh, accessor_t acc) {
            cgh.parallel_for(
                defaultRange,
                parallel_for_range_item_functor<use_offset::no>(acc));
          });

      if (!is_compiler_available(deviceList)) {
        log.note(
            "online compiler is not available -- skipping "
            "test of parallel_for with prebuilt kernel");
      } else {
        {
          auto preBuiltKernel = get_prebuilt_kernel<
              parallel_for_range_item_lambda_prebuilt<use_offset::no>>(queue);

          check_api_call(
              "parallel_for(kernel, range, lambda) with item", log, queue,
              [&](handler &cgh, accessor_t acc) {
                cgh.parallel_for<
                    parallel_for_range_item_lambda_prebuilt<use_offset::no>>(
                    preBuiltKernel, defaultRange,
                    [=](cl::sycl::item<1> item) { acc[item] = item[0]; });
              });
        }

        {
          auto preBuiltKernel = get_prebuilt_kernel<
              parallel_for_range_item_functor<use_offset::no>>(queue);

          check_api_call(
              "parallel_for(kernel, range, functor) with item", log, queue,
              [&](handler &cgh, accessor_t acc) {
                cgh.parallel_for(
                    preBuiltKernel, defaultRange,
                    parallel_for_range_item_functor<use_offset::no>(acc));
              });
        }
      }

      /* parallel_for with offset and item */

      check_api_call(
          "parallel_for(range, id, lambda) with item", log, queue,
          [&](handler &cgh, accessor_t acc) {
            cgh.parallel_for<class parallel_for_range_offset_item_lambda>(
                offsetRange, offset,
                [=](cl::sycl::item<1> item) { acc[item] = item[0]; });
          },
          offset[0], offsetRange[0]);

      check_api_call(
          "parallel_for(range, id, functor) with item", log, queue,
          [&](handler &cgh, accessor_t acc) {
            cgh.parallel_for(
                offsetRange, offset,
                parallel_for_range_item_functor<use_offset::yes>(acc));
          },
          offset[0], offsetRange[0]);

      if (!is_compiler_available(deviceList)) {
        log.note(
            "online compiler is not available -- skipping "
            "test of parallel_for with offset and item with prebuilt kernel");
      } else {
        {
          auto preBuiltKernel = get_prebuilt_kernel<
              parallel_for_range_item_lambda_prebuilt<use_offset::yes>>(queue);

          check_api_call(
              "parallel_for(kernel, range, id, lambda) with item", log, queue,
              [&](handler &cgh, accessor_t acc) {
                cgh.parallel_for<
                    parallel_for_range_item_lambda_prebuilt<use_offset::yes>>(
                    preBuiltKernel, offsetRange, offset,
                    [=](cl::sycl::item<1> item) { acc[item] = item[0]; });
              },
              offset[0], offsetRange[0]);
        }

        {
          auto preBuiltKernel = get_prebuilt_kernel<
              parallel_for_range_item_functor<use_offset::yes>>(queue);

          check_api_call(
              "parallel_for(kernel, range, id, functor) with item", log, queue,
              [&](handler &cgh, accessor_t acc) {
                cgh.parallel_for(
                    preBuiltKernel, offsetRange, offset,
                    parallel_for_range_item_functor<use_offset::yes>(acc));
              },
              offset[0], offsetRange[0]);
        }
      }

      /* parallel_for over nd_range without offset */

      check_api_call("parallel_for(nd_range, lambda)", log, queue,
                     [&](handler &cgh, accessor_t acc) {
                       cgh.parallel_for<class parallel_for_nd_range_lambda>(
                           ndRange, [=](cl::sycl::nd_item<1> ndItem) {
                             acc[ndItem.get_global_id()] =
                                 ndItem.get_global_id(0);
                           });
                     });

      check_api_call("parallel_for(nd_range, functor)", log, queue,
                     [&](handler &cgh, accessor_t acc) {
                       cgh.parallel_for(ndRange,
                                        parallel_for_nd_range_functor(acc));
                     });

      if (!is_compiler_available(deviceList)) {
        log.note(
            "online compiler is not available -- skipping "
            "test of parallel_for over nd_range with prebuilt kernel");
      } else {
        {
          auto preBuiltKernel =
              get_prebuilt_kernel<parallel_for_nd_range_lambda_prebuilt>(queue);

          check_api_call(
              "parallel_for(kernel, nd_range, lambda)", log, queue,
              [&](handler &cgh, accessor_t acc) {
                cgh.parallel_for<parallel_for_nd_range_lambda_prebuilt>(
                    preBuiltKernel, ndRange, [=](cl::sycl::nd_item<1> ndItem) {
                      acc[ndItem.get_global_id()] = ndItem.get_global_id(0);
                    });
              });
        }

        {
          auto preBuiltKernel =
              get_prebuilt_kernel<parallel_for_nd_range_functor>(queue);

          check_api_call("parallel_for(kernel, nd_range, functor)", log, queue,
                         [&](handler &cgh, accessor_t acc) {
                           cgh.parallel_for(preBuiltKernel, ndRange,
                                            parallel_for_nd_range_functor(acc));
                         });
        }
      }

      /* parallel_for over nd_range with offset */

      check_api_call(
          "parallel_for(nd_range, lambda) with offset", log, queue,
          [&](handler &cgh, accessor_t acc) {
            cgh.parallel_for<class parallel_for_nd_range_offset_lambda>(
                offsetNdRange, [=](cl::sycl::nd_item<1> ndItem) {
                  acc[ndItem.get_global_id()] = ndItem.get_global_id(0);
                });
          },
          offset[0], offsetRange[0]);

      check_api_call(
          "parallel_for(nd_range, functor) with offset", log, queue,
          [&](handler &cgh, accessor_t acc) {
            cgh.parallel_for(offsetNdRange, parallel_for_nd_range_functor(acc));
          },
          offset[0], offsetRange[0]);

      if (!is_compiler_available(deviceList)) {
        log.note(
            "online compiler is not available -- skipping "
            "test of parallel_for over nd_range with offset and prebuilt "
            "kernel");
      } else {
        {
          auto preBuiltKernel =
              get_prebuilt_kernel<parallel_for_nd_range_offset_lambda_prebuilt>(
                  queue);

          check_api_call(
              "parallel_for(kernel, nd_range, lambda) with offset", log, queue,
              [&](handler &cgh, accessor_t acc) {
                cgh.parallel_for<parallel_for_nd_range_offset_lambda_prebuilt>(
                    preBuiltKernel, offsetNdRange,
                    [=](cl::sycl::nd_item<1> ndItem) {
                      acc[ndItem.get_global_id()] = ndItem.get_global_id(0);
                    });
              },
              offset[0], offsetRange[0]);
        }

        {
          auto preBuiltKernel =
              get_prebuilt_kernel<parallel_for_nd_range_functor>(queue);

          check_api_call(
              "parallel_for(kernel, nd_range, functor) with offset", log, queue,
              [&](handler &cgh, accessor_t acc) {
                cgh.parallel_for(preBuiltKernel, offsetNdRange,
                                 parallel_for_nd_range_functor(acc));
              },
              offset[0], offsetRange[0]);
        }
      }

      /* parallel_for_work_group (range) */

      check_api_call("parallel_for_work_group(range, lambda)", log, queue,
                     [&](handler &cgh, accessor_t acc) {
                       cgh.parallel_for_work_group<
                           class parallel_for_work_group_1range_lambda>(
                           defaultRange, [=](cl::sycl::group<1> group) {
                             group.parallel_for_work_item(
                                 [&](cl::sycl::h_item<1> item) { acc[0] = 0; });
                             group.parallel_for_work_item(
                                 1, [&](cl::sycl::h_item<1> item) {
                                   if (item.get_global_id(0) > 0) {
                                     acc[item.get_global_id()] =
                                         item.get_global_id(0);
                                   }
                                 });
                           });
                     });

      check_api_call("parallel_for_work_group(range, functor)", log, queue,
                     [&](handler &cgh, accessor_t acc) {
                       cgh.parallel_for_work_group(
                           defaultRange,
                           parallel_for_work_group_dynamic_functor(acc));
                     });

      if (!is_compiler_available(deviceList)) {
        log.note(
            "online compiler is not available -- skipping "
            "test of parallel_for_work_group (range) with prebuilt kernel");
      } else {
        {
          auto preBuiltKernel = get_prebuilt_kernel<
              parallel_for_work_group_1range_lambda_prebuilt>(queue);

          check_api_call(
              "parallel_for_work_group(kernel, range, lambda)", log, queue,
              [&](handler &cgh, accessor_t acc) {
                cgh.parallel_for_work_group<
                    parallel_for_work_group_1range_lambda_prebuilt>(
                    defaultRange, [=](cl::sycl::group<1> group) {
                      group.parallel_for_work_item(
                          [&](cl::sycl::h_item<1> item) { acc[0] = 0; });
                      group.parallel_for_work_item(
                          1, [&](cl::sycl::h_item<1> item) {
                            if (item.get_global_id(0) > 0) {
                              acc[item.get_global_id()] = item.get_global_id(0);
                            }
                          });
                    });
              });
        }

        {
          auto preBuiltKernel =
              get_prebuilt_kernel<parallel_for_work_group_dynamic_functor>(
                  queue);

          check_api_call("parallel_for_work_group(kernel, range, functor)", log,
                         queue, [&](handler &cgh, accessor_t acc) {
                           cgh.parallel_for_work_group(
                               preBuiltKernel, defaultRange,
                               parallel_for_work_group_dynamic_functor(acc));
                         });
        }
      }

      /* parallel_for_work_group (range, range) */

      check_api_call(
          "parallel_for_work_group(range, range, lambda)", log, queue,
          [&](handler &cgh, accessor_t acc) {
            cgh.parallel_for_work_group<
                class parallel_for_work_group_2range_lambda>(
                numWorkGroups, workGroupSize, [=](cl::sycl::group<1> group) {
                  group.parallel_for_work_item([&](cl::sycl::h_item<1> item) {
                    if (item.get_global_id(0) > 0) {
                      acc[item.get_global_id()] = item.get_global_id(0);
                    }
                  });
                  group.parallel_for_work_item(1,
                                               [&](cl::sycl::h_item<1> item) {
                                                 acc[item.get_global_id()] = 0;
                                               });
                });
          });

      check_api_call("parallel_for_work_group(range, range, functor)", log,
                     queue, [&](handler &cgh, accessor_t acc) {
                       cgh.parallel_for_work_group(
                           numWorkGroups, workGroupSize,
                           parallel_for_work_group_fixed_functor(acc));
                     });

      if (!is_compiler_available(deviceList)) {
        log.note(
            "online compiler is not available -- skipping "
            "test of parallel_for_work_group (range, "
            "range) with prebuilt kernel");
      } else {
        {
          auto preBuiltKernel = get_prebuilt_kernel<
              parallel_for_work_group_2range_lambda_prebuilt>(queue);

          check_api_call(
              "parallel_for_work_group(kernel, range, range, lambda)", log,
              queue, [&](handler &cgh, accessor_t acc) {
                cgh.parallel_for_work_group<
                    parallel_for_work_group_2range_lambda_prebuilt>(
                    numWorkGroups, workGroupSize,
                    [=](cl::sycl::group<1> group) {
                      group.parallel_for_work_item(
                          [&](cl::sycl::h_item<1> item) {
                            acc[item.get_global_id()] =
                                item.get_global_id(0) * 2;
                          });
                      group.parallel_for_work_item(
                          defaultRange, [&](cl::sycl::h_item<1> item) {
                            acc[item.get_global_id()] /= 2;
                          });
                    });
              });
        }

        {
          auto preBuiltKernel =
              get_prebuilt_kernel<parallel_for_work_group_fixed_functor>(queue);

          check_api_call(
              "parallel_for_work_group(kernel, range, range, functor)", log,
              queue, [&](handler &cgh, accessor_t acc) {
                cgh.parallel_for_work_group(
                    numWorkGroups, workGroupSize,
                    parallel_for_work_group_fixed_functor(acc));
              });
        }
      }

      /* single_task with kernel object */
      if (!is_compiler_available(deviceList)) {
        log.note(
            "online compiler is not available -- skipping test of "
            "single_task with kernel object");
      } else {
        {
          cl::sycl::program test_program(queue.get_context());
          test_program.build_with_kernel_type<kernel_test_class0>();
          cl::sycl::kernel test_kernel(
              test_program.get_kernel<kernel_test_class0>());

          check_api_call("single_task<kernel_test_class>()", log, queue,
                         [&](handler &cgh, accessor_t acc) {
                           cgh.single_task<kernel_test_class0>([=]() {
                             for (size_t i = 0; i < defaultRange[0]; ++i) {
                               acc[i] = i;
                             }
                           });
                         });
        }
      }

      /* parallel_for with kernel object */
      if (!is_compiler_available(deviceList)) {
        log.note(
            "online compiler is not available -- skipping test of "
            "parallel_for with kernel object");
      } else {
        {
          cl::sycl::program test_program(queue.get_context());
          test_program.build_with_kernel_type<kernel_test_class1>();
          cl::sycl::kernel test_kernel(
              test_program.get_kernel<kernel_test_class1>());

          check_api_call("parallel_for(range, kernel) with id", log, queue,
                         [&](handler &cgh, accessor_t acc) {
                           cgh.parallel_for<kernel_test_class1>(
                               defaultRange,
                               [=](cl::sycl::id<1> id) { acc[id] = id[0]; });
                         });
        }

        {
          cl::sycl::program test_program(queue.get_context());
          test_program.build_with_kernel_type<kernel_test_class2>();
          cl::sycl::kernel test_kernel(
              test_program.get_kernel<kernel_test_class2>());

          check_api_call(
              "parallel_for(range, offset, kernel) with id", log, queue,
              [&](handler &cgh, accessor_t acc) {
                cgh.parallel_for<kernel_test_class2>(
                    offsetRange, offset,
                    [=](cl::sycl::id<1> id) { acc[id] = id[0]; });
              },
              offset[0], offsetRange[0]);
        }

        {
          cl::sycl::program test_program(queue.get_context());
          test_program.build_with_kernel_type<kernel_test_class3>();
          cl::sycl::kernel test_kernel(
              test_program.get_kernel<kernel_test_class3>());

          check_api_call("parallel_for(range, kernel) with item", log, queue,
                         [&](handler &cgh, accessor_t acc) {
                           cgh.parallel_for<kernel_test_class3>(
                               defaultRange, [=](cl::sycl::item<1> item) {
                                 acc[item] = item[0];
                               });
                         });
        }

        {
          cl::sycl::program test_program(queue.get_context());
          test_program.build_with_kernel_type<kernel_test_class4>();
          cl::sycl::kernel test_kernel(
              test_program.get_kernel<kernel_test_class4>());

          check_api_call(
              "parallel_for(range, offset, kernel) with item", log, queue,
              [&](handler &cgh, accessor_t acc) {
                cgh.parallel_for<kernel_test_class4>(
                    offsetRange, offset,
                    [=](cl::sycl::item<1> item) { acc[item] = item[0]; });
              },
              offset[0], offsetRange[0]);
        }

        {
          cl::sycl::program test_program(queue.get_context());
          test_program.build_with_kernel_type<kernel_test_class5>();
          cl::sycl::kernel test_kernel(
              test_program.get_kernel<kernel_test_class5>());

          check_api_call("parallel_for(nd_range, kernel);", log, queue,
                         [&](handler &cgh, accessor_t acc) {
                           cgh.parallel_for<kernel_test_class5>(
                               ndRange, [=](cl::sycl::nd_item<1> ndItem) {
                                 acc[ndItem.get_global_id()] =
                                     ndItem.get_global_id(0);
                               });
                         });
        }
      }

    } catch (const cl::sycl::exception &e) {
      log_exception(log, e);
      FAIL(log,
           "A SYCL exception was "
           "caught");
    }
  };
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace handler_invoke_api__ */

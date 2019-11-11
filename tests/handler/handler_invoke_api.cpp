/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
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
  single_task_functor(accessor_t acc) : acc(acc) {}

  void operator()() { acc[0] = 10; }

  accessor_t acc;
};

template <int useOffset>
struct parallel_for_range_id_functor {
  parallel_for_range_id_functor(accessor_t acc) : acc(acc) {}

  void operator()(cl::sycl::id<1> id) { acc[0] = 10; }

  accessor_t acc;
};

template <int useOffset>
struct parallel_for_range_item_functor {
  parallel_for_range_item_functor(accessor_t acc) : acc(acc) {}

  void operator()(cl::sycl::item<1> item) { acc[0] = 10; }

  accessor_t acc;
};

struct parallel_for_nd_range_functor {
  parallel_for_nd_range_functor(accessor_t acc) : acc(acc) {}

  void operator()(cl::sycl::nd_item<1> ndItem) { acc[0] = 10; }

  accessor_t acc;
};

template <int>
struct parallel_for_work_group_range_functor {
  parallel_for_work_group_range_functor(accessor_t acc) : acc(acc) {}

  void operator()(cl::sycl::group<1> group) {
    group.parallel_for_work_item(
        [&](cl::sycl::h_item<1> item) { acc[0] = 10; });

    cl::sycl::range<1> subRange(1);

    group.parallel_for_work_item(
        subRange, [&](cl::sycl::h_item<1> item) { acc[0] = 10; });
  }

  accessor_t acc;
};
using parallel_for_work_group_1range_functor =
    parallel_for_work_group_range_functor<1>;
using parallel_for_work_group_2range_functor =
    parallel_for_work_group_range_functor<2>;

class single_task_lambda_prebuilt;
template <int useOffset>
class parallel_for_range_id_lambda_prebuilt;
template <int useOffset>
class parallel_for_range_item_lambda_prebuilt;
class parallel_for_nd_range_lambda_prebuilt;
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
class parallel_for_work_group_1range_lambda;
class parallel_for_work_group_2range_lambda;

/** tests the invoke APIs
 */
class TEST_NAME : public sycl_cts::util::test_base {
 public:
  const cl::sycl::range<1> range = 1;

  /**
   * @brief Checks that the handler API call works correctly
   * @tparam kernel_wrapper Type of the lambda used to invoke the API method
   * @param methodName Name of the method to be checked
   * @param log Test logger object
   * @param queue Queue to submit the kernel to
   * @param kernelWrapper A lambda that contains a call to a handler API method.
   *
   * The lambda is passed a handler and a read_write accessor. The first element
   * of the accessor must be assigned the value 10.
   */
  template <class kernel_wrapper>
  void check_api_call(cl::sycl::string_class methodName, util::logger &log,
                      cl::sycl::queue &queue, kernel_wrapper &&kernelWrapper) {
    log.note("Check %s", methodName.c_str());
    int result = 0;
    {
      auto buf = cl::sycl::buffer<int, 1>(&result, range);
      queue.submit([&](cl::sycl::handler &cgh) {
        auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
        kernelWrapper(cgh, acc);
      });
    }
    CHECK_VALUE_SCALAR(log, result, 10);
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
      const cl::sycl::id<1> offset(0);
      const cl::sycl::nd_range<1> ndRange(range, range);
      const cl::sycl::range<1> globalRange(2);
      const cl::sycl::range<1> localRange(2);
      const cl::sycl::range<1> subRange(1);

      /* single_task */

      check_api_call(
          "single_task(lambda)", log, queue, [&](handler &cgh, accessor_t acc) {
            cgh.single_task<class single_task_lambda>([=]() { acc[0] = 10; });
          });

      check_api_call("single_task(functor)", log, queue,
                     [&](handler &cgh, accessor_t acc) {
                       cgh.single_task(single_task_functor(acc));
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
                               preBuiltKernel, [=]() { acc[0] = 10; });
                         });
        }

        {
          auto preBuiltKernel = get_prebuilt_kernel<single_task_functor>(queue);

          check_api_call("single_task(kernel, functor)", log, queue,
                         [&](handler &cgh, accessor_t acc) {
                           cgh.single_task<single_task_functor>(
                               preBuiltKernel, single_task_functor(acc));
                         });
        }
      }

      /* parallel_for with id */

      check_api_call("parallel_for(range, lambda) with id", log, queue,
                     [&](handler &cgh, accessor_t acc) {
                       cgh.parallel_for<class parallel_for_range_id_lambda>(
                           range, [=](cl::sycl::id<1> id) { acc[0] = 10; });
                     });

      check_api_call("parallel_for(range, functor) with id", log, queue,
                     [&](handler &cgh, accessor_t acc) {
                       cgh.parallel_for(
                           range,
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
                    preBuiltKernel, range,
                    [=](cl::sycl::id<1> id) { acc[0] = 10; });
              });
        }

        {
          auto preBuiltKernel = get_prebuilt_kernel<
              parallel_for_range_id_functor<use_offset::no>>(queue);

          check_api_call(
              "parallel_for(kernel, range, functor) with id", log, queue,
              [&](handler &cgh, accessor_t acc) {
                cgh.parallel_for(
                    preBuiltKernel, range,
                    parallel_for_range_id_functor<use_offset::no>(acc));
              });
        }
      }

      /* parallel_for with offset and id */

      check_api_call(
          "parallel_for(range, id, lambda) with id", log, queue,
          [&](handler &cgh, accessor_t acc) {
            cgh.parallel_for<class parallel_for_range_offset_id_lambda>(
                range, offset, [=](cl::sycl::id<1> id) { acc[0] = 10; });
          });

      check_api_call("parallel_for(range, id, functor) with id", log, queue,
                     [&](handler &cgh, accessor_t acc) {
                       cgh.parallel_for(
                           range, offset,
                           parallel_for_range_id_functor<use_offset::yes>(acc));
                     });

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
                    preBuiltKernel, range, offset,
                    [=](cl::sycl::id<1> id) { acc[0] = 10; });
              });
        }

        {
          auto preBuiltKernel = get_prebuilt_kernel<
              parallel_for_range_id_functor<use_offset::yes>>(queue);

          check_api_call(
              "parallel_for(kernel, range, id, functor) with id", log, queue,
              [&](handler &cgh, accessor_t acc) {
                cgh.parallel_for(
                    preBuiltKernel, range, offset,
                    parallel_for_range_id_functor<use_offset::yes>(acc));
              });
        }
      }

      /* parallel_for with item */

      check_api_call("parallel_for(range, lambda) with item", log, queue,
                     [&](handler &cgh, accessor_t acc) {
                       cgh.parallel_for<class parallel_for_range_item_lambda>(
                           range, [=](cl::sycl::item<1> item) { acc[0] = 10; });
                     });

      check_api_call(
          "parallel_for(range, functor) with item", log, queue,
          [&](handler &cgh, accessor_t acc) {
            cgh.parallel_for(
                range, parallel_for_range_item_functor<use_offset::no>(acc));
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
                    preBuiltKernel, range,
                    [=](cl::sycl::item<1> item) { acc[0] = 10; });
              });
        }

        {
          auto preBuiltKernel = get_prebuilt_kernel<
              parallel_for_range_item_functor<use_offset::no>>(queue);

          check_api_call(
              "parallel_for(kernel, range, functor) with item", log, queue,
              [&](handler &cgh, accessor_t acc) {
                cgh.parallel_for(
                    preBuiltKernel, range,
                    parallel_for_range_item_functor<use_offset::no>(acc));
              });
        }
      }

      /* parallel_for with offset and item */

      check_api_call(
          "parallel_for(range, id, lambda) with item", log, queue,
          [&](handler &cgh, accessor_t acc) {
            cgh.parallel_for<class parallel_for_range_offset_item_lambda>(
                range, offset, [=](cl::sycl::item<1> item) { acc[0] = 10; });
          });

      check_api_call(
          "parallel_for(range, id, functor) with item", log, queue,
          [&](handler &cgh, accessor_t acc) {
            cgh.parallel_for(
                range, offset,
                parallel_for_range_item_functor<use_offset::yes>(acc));
          });

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
                    preBuiltKernel, range, offset,
                    [=](cl::sycl::item<1> item) { acc[0] = 10; });
              });
        }

        {
          auto preBuiltKernel = get_prebuilt_kernel<
              parallel_for_range_item_functor<use_offset::yes>>(queue);

          check_api_call(
              "parallel_for(kernel, range, id, functor) with item", log, queue,
              [&](handler &cgh, accessor_t acc) {
                cgh.parallel_for(
                    preBuiltKernel, range, offset,
                    parallel_for_range_item_functor<use_offset::yes>(acc));
              });
        }
      }

      /* parallel_for over nd_range */

      check_api_call("parallel_for(nd_range, lambda)", log, queue,
                     [&](handler &cgh, accessor_t acc) {
                       cgh.parallel_for<class parallel_for_nd_range_lambda>(
                           ndRange,
                           [=](cl::sycl::nd_item<1> ndItem) { acc[0] = 10; });
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
                    preBuiltKernel, ndRange,
                    [=](cl::sycl::nd_item<1> ndItem) { acc[0] = 10; });
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

      /* parallel_for_work_group (range) */

      {
        const cl::sycl::range<1> range = 1;
        check_api_call(
            "parallel_for_work_group(range, lambda)", log, queue,
            [&](handler &cgh, accessor_t acc) {
              cgh.parallel_for_work_group<
                  class parallel_for_work_group_1range_lambda>(
                  range, [=](cl::sycl::group<1> group) {
                    group.parallel_for_work_item(
                        [&](cl::sycl::h_item<1> item) { acc[0] = 10; });
                    group.parallel_for_work_item(
                        subRange,
                        [&](cl::sycl::h_item<1> item) { acc[0] = 10; });
                  });
            });
      }

      check_api_call("parallel_for_work_group(range, functor)", log, queue,
                     [&](handler &cgh, accessor_t acc) {
                       cgh.parallel_for_work_group(
                           range, parallel_for_work_group_1range_functor(acc));
                     });

      if (!is_compiler_available(deviceList)) {
        log.note(
            "online compiler is not available -- skipping "
            "test of parallel_for_work_group (range) with prebuilt kernel");
      } else {
        {
          const cl::sycl::range<1> range = 1;
          auto preBuiltKernel = get_prebuilt_kernel<
              parallel_for_work_group_1range_lambda_prebuilt>(queue);

          check_api_call(
              "parallel_for_work_group(kernel, range, lambda)", log, queue,
              [&](handler &cgh, accessor_t acc) {
                cgh.parallel_for_work_group<
                    parallel_for_work_group_1range_lambda_prebuilt>(
                    range, [=](cl::sycl::group<1> group) {
                      group.parallel_for_work_item(
                          [&](cl::sycl::h_item<1> item) { acc[0] = 10; });
                      group.parallel_for_work_item(
                          subRange,
                          [&](cl::sycl::h_item<1> item) { acc[0] = 10; });
                    });
              });
        }

        {
          auto preBuiltKernel =
              get_prebuilt_kernel<parallel_for_work_group_1range_functor>(
                  queue);

          check_api_call("parallel_for_work_group(kernel, range, functor)", log,
                         queue, [&](handler &cgh, accessor_t acc) {
                           cgh.parallel_for_work_group(
                               preBuiltKernel, range,
                               parallel_for_work_group_1range_functor(acc));
                         });
        }
      }

      /* parallel_for_work_group (range, range) */

      check_api_call(
          "parallel_for_work_group(range, range, lambda)", log, queue,
          [&](handler &cgh, accessor_t acc) {
            cgh.parallel_for_work_group<
                class parallel_for_work_group_2range_lambda>(
                globalRange, localRange, [=](cl::sycl::group<1> group) {
                  group.parallel_for_work_item(
                      [&](cl::sycl::h_item<1> item) { acc[0] = 10; });
                  group.parallel_for_work_item(
                      subRange, [&](cl::sycl::h_item<1> item) { acc[0] = 10; });
                });
          });

      check_api_call("parallel_for_work_group(range, range, functor)", log,
                     queue, [&](handler &cgh, accessor_t acc) {
                       cgh.parallel_for_work_group(
                           globalRange, localRange,
                           parallel_for_work_group_2range_functor(acc));
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
                    globalRange, localRange, [=](cl::sycl::group<1> group) {
                      group.parallel_for_work_item(
                          [&](cl::sycl::h_item<1> item) { acc[0] = 10; });
                      group.parallel_for_work_item(
                          subRange,
                          [&](cl::sycl::h_item<1> item) { acc[0] = 10; });
                    });
              });
        }

        {
          auto preBuiltKernel =
              get_prebuilt_kernel<parallel_for_work_group_2range_functor>(
                  queue);

          check_api_call(
              "parallel_for_work_group(kernel, range, range, functor)", log,
              queue, [&](handler &cgh, accessor_t acc) {
                cgh.parallel_for_work_group(
                    globalRange, localRange,
                    parallel_for_work_group_2range_functor(acc));
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

          check_api_call(
              "single_task<kernel_test_class>()", log, queue,
              [&](handler &cgh, accessor_t acc) {
                cgh.single_task<kernel_test_class0>([=]() { acc[0] = 10; });
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
                               range, [=](cl::sycl::id<1> id) { acc[0] = 10; });
                         });
        }

        {
          cl::sycl::program test_program(queue.get_context());
          test_program.build_with_kernel_type<kernel_test_class2>();
          cl::sycl::kernel test_kernel(
              test_program.get_kernel<kernel_test_class2>());

          check_api_call("parallel_for(range, offset, kernel) with id", log,
                         queue, [&](handler &cgh, accessor_t acc) {
                           cgh.parallel_for<kernel_test_class2>(
                               range, offset,
                               [=](cl::sycl::id<1> id) { acc[0] = 10; });
                         });
        }

        {
          cl::sycl::program test_program(queue.get_context());
          test_program.build_with_kernel_type<kernel_test_class3>();
          cl::sycl::kernel test_kernel(
              test_program.get_kernel<kernel_test_class3>());

          check_api_call("parallel_for(range, kernel) with item", log, queue,
                         [&](handler &cgh, accessor_t acc) {
                           cgh.parallel_for<kernel_test_class3>(
                               range,
                               [=](cl::sycl::item<1> item) { acc[0] = 10; });
                         });
        }

        {
          cl::sycl::program test_program(queue.get_context());
          test_program.build_with_kernel_type<kernel_test_class4>();
          cl::sycl::kernel test_kernel(
              test_program.get_kernel<kernel_test_class4>());

          check_api_call("parallel_for(range, offset, kernel) with item", log,
                         queue, [&](handler &cgh, accessor_t acc) {
                           cgh.parallel_for<kernel_test_class4>(
                               range, offset,
                               [=](cl::sycl::item<1> item) { acc[0] = 10; });
                         });
        }

        {
          cl::sycl::program test_program(queue.get_context());
          test_program.build_with_kernel_type<kernel_test_class5>();
          cl::sycl::kernel test_kernel(
              test_program.get_kernel<kernel_test_class5>());

          check_api_call(
              "parallel_for(nd_range, kernel);", log, queue,
              [&](handler &cgh, accessor_t acc) {
                cgh.parallel_for<kernel_test_class5>(
                    ndRange, [=](cl::sycl::nd_item<1> ndItem) { acc[0] = 10; });
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

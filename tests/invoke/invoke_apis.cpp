/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME invoke_apis

namespace invoke_apis__ {
using namespace sycl_cts;

struct single_task_functor {
  single_task_functor(
      cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::global_buffer> acc)
      : accessor(acc) {}

  void operator()() { accessor[0] = 10; }

  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                     cl::sycl::access::target::global_buffer> accessor;
};

struct parallel_for_nd_range_functor {
  parallel_for_nd_range_functor(
      cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::global_buffer> acc)
      : accessor(acc) {}

  void operator()(cl::sycl::nd_item<1> ndItem) { accessor[0] = 10; }

  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                     cl::sycl::access::target::global_buffer> accessor;
};

struct parallel_for_range_item_functor {
  parallel_for_range_item_functor(
      cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::global_buffer> acc)
      : accessor(acc) {}

  void operator()(cl::sycl::item<1> item) { accessor[0] = 10; }

  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                     cl::sycl::access::target::global_buffer> accessor;
};

struct parallel_for_range_id_functor {
  parallel_for_range_id_functor(
      cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::global_buffer> acc)
      : accessor(acc) {}

  void operator()(cl::sycl::id<1> id) { accessor[0] = 10; }

  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                     cl::sycl::access::target::global_buffer> accessor;
};

struct parallel_for_work_group_1range_functor {
  parallel_for_work_group_1range_functor(
      cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::global_buffer> acc)
      : accessor(acc) {}

  void operator()(cl::sycl::group<1> group) {
    parallel_for_work_item(group,
                           [&](cl::sycl::item<1> item) { accessor[0] = 10; });

    cl::sycl::range<1> subRange(1);

    parallel_for_work_item(group, subRange,
                           [&](cl::sycl::item<1> item) { accessor[0] = 10; });
  }

  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                     cl::sycl::access::target::global_buffer> accessor;
};

class single_task_lambda_prebuilt;
class parallel_for_nd_range_lambda_prebuilt;
class parallel_for_range_item_lambda_prebuilt;
class parallel_for_range_id_lambda_prebuilt;
class parallel_for_work_group_1range_lambda_prebuilt;
class parallel_for_work_group_2range_lambda_prebuilt;

/** tests the invoke apis
 */
class TEST_NAME : public sycl_cts::util::test_base {
 public:
  /** return information about this test
   */
  virtual void get_info(test_base::info& out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute this test
   */
  virtual void run(util::logger& log) override {
    try {
      cts_selector selector;
      cl::sycl::queue queue(selector);

      /** check single_task(lambda) function
      */
      {
        int result = 0;

        {
          cl::sycl::buffer<int, 1> buffer(&result, cl::sycl::range<1>(1));

          queue.submit([&](cl::sycl::handler& handler) {
            cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                               cl::sycl::access::target::global_buffer>
                accessor(buffer, handler);

            handler.single_task<class single_task_lambda>(
                [=]() { accessor[0] = 10; });
          });
        }

        if (result != 10) {
          FAIL(log, "single_task(lambda) failed to execute correctly");
        }
      }

      /** check single_task(functor) function
      */
      {
        int result = 0;

        {
          cl::sycl::buffer<int, 1> buffer(&result, cl::sycl::range<1>(1));

          queue.submit([&](cl::sycl::handler& handler) {
            cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                               cl::sycl::access::target::global_buffer>
                accessor(buffer, handler);

            handler.single_task(single_task_functor(accessor));
          });
        }

        if (result != 10) {
          FAIL(log, "single_task(functor) failed to execute correctly");
        }
      }

      /** check single_task(kernel, lambda) function
      */
      {
        int result = 0;

        cl::sycl::program program(queue.get_context());
        cl::sycl::kernel preBuiltKernel =
            program.get_kernel<single_task_lambda_prebuilt>();

        {
          cl::sycl::buffer<int, 1> buffer(&result, cl::sycl::range<1>(1));

          queue.submit([&](cl::sycl::handler& handler) {
            cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                               cl::sycl::access::target::global_buffer>
                accessor(buffer, handler);

            handler.single_task<single_task_lambda_prebuilt>(
                preBuiltKernel, [=]() { accessor[0] = 10; });
          });
        }

        if (result != 10) {
          FAIL(log, "single_task(kernel, lambda) failed to execute correctly");
        }
      }

      /** check single_task(kernel, functor) function
      */
      {
        int result = 0;

        cl::sycl::program program(queue.get_context());
        cl::sycl::kernel preBuiltKernel =
            program.get_kernel<single_task_functor>();

        {
          cl::sycl::buffer<int, 1> buffer(&result, cl::sycl::range<1>(1));

          queue.submit([&](cl::sycl::handler& handler) {
            cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                               cl::sycl::access::target::global_buffer>
                accessor(buffer, handler);

            handler.single_task<single_task_functor>(
                preBuiltKernel, single_task_functor(accessor));
          });
        }

        if (result != 10) {
          FAIL(log, "single_task(kernel, functor) failed to execute correctly");
        }
      }

      /** check parallel_for(nd_range, lambda) function
      */
      {
        int result = 0;

        {
          cl::sycl::buffer<int, 1> buffer(&result, cl::sycl::range<1>(1));

          queue.submit([&](cl::sycl::handler& handler) {
            cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                               cl::sycl::access::target::global_buffer>
                accessor(buffer, handler);

            cl::sycl::nd_range<1> ndRange(cl::sycl::range<1>(1),
                                          cl::sycl::range<1>(1));

            handler.parallel_for<class parallel_for_nd_range_lambda>(
                ndRange,
                [=](cl::sycl::nd_item<1> ndItem) { accessor[0] = 10; });
          });
        }

        if (result != 10) {
          FAIL(log,
               "parallel_for(nd_range, lambda) failed to execute correctly");
        }
      }

      /** check parallel_for(nd_range, functor) function
      */
      {
        int result = 0;

        {
          cl::sycl::buffer<int, 1> buffer(&result, cl::sycl::range<1>(1));

          queue.submit([&](cl::sycl::handler& handler) {
            cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                               cl::sycl::access::target::global_buffer>
                accessor(buffer, handler);

            cl::sycl::nd_range<1> ndRange(cl::sycl::range<1>(1),
                                          cl::sycl::range<1>(1));

            handler.parallel_for(ndRange,
                                 parallel_for_nd_range_functor(accessor));
          });
        }

        if (result != 10) {
          FAIL(log,
               "parallel_for(nd_range, functor) failed to execute correctly");
        }
      }

      /** check parallel_for(kernel, nd_range, lambda) function
      */
      {
        int result = 0;

        cl::sycl::program program(queue.get_context());
        cl::sycl::kernel preBuiltKernel =
            program.get_kernel<parallel_for_nd_range_lambda_prebuilt>();

        {
          cl::sycl::buffer<int, 1> buffer(&result, cl::sycl::range<1>(1));

          queue.submit([&](cl::sycl::handler& handler) {
            cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                               cl::sycl::access::target::global_buffer>
                accessor(buffer, handler);

            cl::sycl::nd_range<1> ndRange(cl::sycl::range<1>(1),
                                          cl::sycl::range<1>(1));

            handler.parallel_for<parallel_for_nd_range_lambda_prebuilt>(
                preBuiltKernel, ndRange,
                [=](cl::sycl::nd_item<1> ndItem) { accessor[0] = 10; });
          });
        }

        if (result != 10) {
          FAIL(log,
               "parallel_for(kernel, nd_range, lambda) failed to execute "
               "correctly");
        }
      }

      /** check parallel_for(kernel, nd_range, functor) function
      */
      {
        int result = 0;

        cl::sycl::program program(queue.get_context());
        cl::sycl::kernel preBuiltKernel =
            program.get_kernel<parallel_for_nd_range_functor>();

        {
          cl::sycl::buffer<int, 1> buffer(&result, cl::sycl::range<1>(1));

          queue.submit([&](cl::sycl::handler& handler) {
            cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                               cl::sycl::access::target::global_buffer>
                accessor(buffer, handler);

            cl::sycl::nd_range<1> ndRange(cl::sycl::range<1>(1),
                                          cl::sycl::range<1>(1));

            handler.parallel_for(preBuiltKernel, ndRange,
                                 parallel_for_nd_range_functor(accessor));
          });
        }

        if (result != 10) {
          FAIL(log,
               "parallel_for(kernel, nd_range, functor) failed to execute "
               "correctly");
        }
      }

      /** check parallel_for(range, lambda) w/ item function
      */
      {
        int result = 0;

        {
          cl::sycl::buffer<int, 1> buffer(&result, cl::sycl::range<1>(1));

          queue.submit([&](cl::sycl::handler& handler) {
            cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                               cl::sycl::access::target::global_buffer>
                accessor(buffer, handler);

            cl::sycl::range<1> range(1);

            handler.parallel_for<class parallel_for_range_item_lambda>(
                range, [=](cl::sycl::item<1> item) { accessor[0] = 10; });
          });
        }

        if (result != 10) {
          FAIL(log,
               "parallel_for(range, lambda) w/ item failed to execute "
               "correctly");
        }
      }

      /** check parallel_for(range, functor) w/ item function
      */
      {
        int result = 0;

        {
          cl::sycl::buffer<int, 1> buffer(&result, cl::sycl::range<1>(1));

          queue.submit([&](cl::sycl::handler& handler) {
            cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                               cl::sycl::access::target::global_buffer>
                accessor(buffer, handler);

            cl::sycl::range<1> range(1);

            handler.parallel_for(range,
                                 parallel_for_range_item_functor(accessor));
          });
        }

        if (result != 10) {
          FAIL(log,
               "parallel_for(range, functor) w/ item failed to execute "
               "correctly");
        }
      }

      /** check parallel_for(kernel, range, lambda) w/ item function
      */
      {
        int result = 0;

        cl::sycl::program program(queue.get_context());
        cl::sycl::kernel preBuiltKernel =
            program.get_kernel<parallel_for_range_item_lambda_prebuilt>();

        {
          cl::sycl::buffer<int, 1> buffer(&result, cl::sycl::range<1>(1));

          queue.submit([&](cl::sycl::handler& handler) {
            cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                               cl::sycl::access::target::global_buffer>
                accessor(buffer, handler);

            cl::sycl::range<1> range(1);

            handler.parallel_for<parallel_for_range_item_lambda_prebuilt>(
                preBuiltKernel, range,
                [=](cl::sycl::item<1> item) { accessor[0] = 10; });
          });
        }

        if (result != 10) {
          FAIL(log,
               "parallel_for(kernel, range, lambda) w/ item failed to execute "
               "correctly");
        }
      }

      /** check parallel_for(kernel, range, functor) w/ item function
      */
      {
        int result = 0;

        cl::sycl::program program(queue.get_context());
        cl::sycl::kernel preBuiltKernel =
            program.get_kernel<parallel_for_range_item_functor>();

        {
          cl::sycl::buffer<int, 1> buffer(&result, cl::sycl::range<1>(1));

          queue.submit([&](cl::sycl::handler& handler) {
            cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                               cl::sycl::access::target::global_buffer>
                accessor(buffer, handler);

            cl::sycl::range<1> range(1);

            handler.parallel_for(preBuiltKernel, range,
                                 parallel_for_range_item_functor(accessor));
          });
        }

        if (result != 10) {
          FAIL(log,
               "parallel_for(kernel, range, functor) w/ item failed to execute "
               "correctly");
        }
      }

      /** check parallel_for(range, lambda) w/ id function
      */
      {
        int result = 0;

        {
          cl::sycl::buffer<int, 1> buffer(&result, cl::sycl::range<1>(1));

          queue.submit([&](cl::sycl::handler& handler) {
            cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                               cl::sycl::access::target::global_buffer>
                accessor(buffer, handler);

            cl::sycl::range<1> range(1);

            handler.parallel_for<class parallel_for_range_id_lambda>(
                range, [=](cl::sycl::id<1> id) { accessor[0] = 10; });
          });
        }

        if (result != 10) {
          FAIL(log,
               "parallel_for(range, lambda) w/ id failed to execute correctly");
        }
      }

      /** check parallel_for(range, functor) w/ id function
      */
      {
        int result = 0;

        {
          cl::sycl::buffer<int, 1> buffer(&result, cl::sycl::range<1>(1));

          queue.submit([&](cl::sycl::handler& handler) {
            cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                               cl::sycl::access::target::global_buffer>
                accessor(buffer, handler);

            cl::sycl::range<1> range(1);

            handler.parallel_for(range,
                                 parallel_for_range_id_functor(accessor));
          });
        }

        if (result != 10) {
          FAIL(
              log,
              "parallel_for(range, functor) w/ id failed to execute correctly");
        }
      }

      /** check parallel_for(kernel, range, lambda) w/ id function
      */
      {
        int result = 0;

        cl::sycl::program program(queue.get_context());
        cl::sycl::kernel preBuiltKernel =
            program.get_kernel<parallel_for_range_id_lambda_prebuilt>();

        {
          cl::sycl::buffer<int, 1> buffer(&result, cl::sycl::range<1>(1));

          queue.submit([&](cl::sycl::handler& handler) {
            cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                               cl::sycl::access::target::global_buffer>
                accessor(buffer, handler);

            cl::sycl::range<1> range(1);

            handler.parallel_for<parallel_for_range_id_lambda_prebuilt>(
                preBuiltKernel, range,
                [=](cl::sycl::id<1> id) { accessor[0] = 10; });
          });
        }

        if (result != 10) {
          FAIL(log,
               "parallel_for(kernel, range, lambda) w/ id failed to execute "
               "correctly");
        }
      }

      /** check parallel_for(kernel, range, functor) w/ id function
      */
      {
        int result = 0;

        cl::sycl::program program(queue.get_context());
        cl::sycl::kernel preBuiltKernel =
            program.get_kernel<parallel_for_range_id_functor>();

        {
          cl::sycl::buffer<int, 1> buffer(&result, cl::sycl::range<1>(1));

          queue.submit([&](cl::sycl::handler& handler) {
            cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                               cl::sycl::access::target::global_buffer>
                accessor(buffer, handler);

            cl::sycl::range<1> range(1);

            handler.parallel_for(preBuiltKernel, range,
                                 parallel_for_range_id_functor(accessor));
          });
        }

        if (result != 10) {
          FAIL(log,
               "parallel_for(kernel, range, functor) w/ id failed to execute "
               "correctly");
        }
      }

      /** check parallel_for_work_group(range, lambda) function
      */
      {
        int result = 0;

        {
          cl::sycl::buffer<int, 1> buffer(&result, cl::sycl::range<1>(1));

          queue.submit([&](cl::sycl::handler& handler) {
            cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                               cl::sycl::access::target::global_buffer>
                accessor(buffer, handler);

            cl::sycl::range<1> range(2);

            handler.parallel_for_work_group<
                class parallel_for_work_group_1range_lambda>(
                range, [=](cl::sycl::group<1> group) {
                  parallel_for_work_item(
                      group, [&](cl::sycl::item<1> item) { accessor[0] = 10; });

                  cl::sycl::range<1> subRange(1);

                  parallel_for_work_item(
                      group, subRange,
                      [&](cl::sycl::item<1> item) { accessor[0] = 10; });
                });
          });
        }

        if (result != 10) {
          FAIL(log,
               "parallel_for_work_group(range, lambda) failed to execute "
               "correctly");
        }
      }

      /** check parallel_for_work_group(range, functor) function
      */
      {
        int result = 0;

        {
          cl::sycl::buffer<int, 1> buffer(&result, cl::sycl::range<1>(1));

          queue.submit([&](cl::sycl::handler& handler) {
            cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                               cl::sycl::access::target::global_buffer>
                accessor(buffer, handler);

            cl::sycl::range<1> range(2);

            handler.parallel_for_work_group(
                range, parallel_for_work_group_1range_functor(accessor));
          });
        }

        if (result != 10) {
          FAIL(log,
               "parallel_for_work_group(range, functor) failed to execute "
               "correctly");
        }
      }

      /** check parallel_for_work_group(kernel, range, lambda) function
      */
      {
        int result = 0;

        cl::sycl::program program(queue.get_context());
        cl::sycl::kernel preBuiltKernel =
            program
                .get_kernel<parallel_for_work_group_1range_lambda_prebuilt>();

        {
          cl::sycl::buffer<int, 1> buffer(&result, cl::sycl::range<1>(1));

          queue.submit([&](cl::sycl::handler& handler) {
            cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                               cl::sycl::access::target::global_buffer>
                accessor(buffer, handler);

            cl::sycl::range<1> range(2);

            handler.parallel_for_work_group<
                parallel_for_work_group_1range_lambda_prebuilt>(
                preBuiltKernel, range, [=](cl::sycl::group<1> group) {
                  parallel_for_work_item(
                      group, [&](cl::sycl::item<1> item) { accessor[0] = 10; });

                  cl::sycl::range<1> subRange(1);

                  parallel_for_work_item(
                      group, subRange,
                      [&](cl::sycl::item<1> item) { accessor[0] = 10; });
                });
          });
        }

        if (result != 10) {
          FAIL(log,
               "parallel_for_work_group(range, lambda) failed to execute "
               "correctly");
        }
      }

      /** check parallel_for_work_group(kernel, range, functor) function
      */
      {
        int result = 0;

        cl::sycl::program program(queue.get_context());
        cl::sycl::kernel preBuiltKernel =
            program.get_kernel<parallel_for_work_group_1range_functor>();

        {
          cl::sycl::buffer<int, 1> buffer(&result, cl::sycl::range<1>(1));

          queue.submit([&](cl::sycl::handler& handler) {
            cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                               cl::sycl::access::target::global_buffer>
                accessor(buffer, handler);

            cl::sycl::range<1> range(2);

            handler.parallel_for_work_group(
                preBuiltKernel, range,
                parallel_for_work_group_1range_functor(accessor));
          });
        }

        if (result != 10) {
          FAIL(log,
               "parallel_for_work_group(range, functor) failed to execute "
               "correctly");
        }
      }

      /** check parallel_for_work_group(range, range, lambda) function
      */
      {
        int result = 0;

        {
          cl::sycl::buffer<int, 1> buffer(&result, cl::sycl::range<1>(1));

          queue.submit([&](cl::sycl::handler& handler) {
            cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                               cl::sycl::access::target::global_buffer>
                accessor(buffer, handler);

            cl::sycl::range<1> globalRange(2);
            cl::sycl::range<1> localRange(2);

            handler.parallel_for_work_group<
                class parallel_for_work_group_1range_lambda>(
                globalRange, localRange, [=](cl::sycl::group<1> group) {
                  parallel_for_work_item(
                      group, [&](cl::sycl::item<1> item) { accessor[0] = 10; });

                  cl::sycl::range<1> subRange(1);

                  parallel_for_work_item(
                      group, subRange,
                      [&](cl::sycl::item<1> item) { accessor[0] = 10; });
                });
          });
        }

        if (result != 10) {
          FAIL(log,
               "parallel_for_work_group(range, range, lambda) failed to "
               "execute correctly");
        }
      }

      /** check parallel_for_work_group(range, range, functor) function
      */
      {
        int result = 0;

        {
          cl::sycl::buffer<int, 1> buffer(&result, cl::sycl::range<1>(1));

          queue.submit([&](cl::sycl::handler& handler) {
            cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                               cl::sycl::access::target::global_buffer>
                accessor(buffer, handler);

            cl::sycl::range<1> globalRange(2);
            cl::sycl::range<1> localRange(2);

            handler.parallel_for_work_group(
                globalRange, localRange,
                parallel_for_work_group_1range_functor(accessor));
          });
        }

        if (result != 10) {
          FAIL(log,
               "parallel_for_work_group(range, range, functor) failed to "
               "execute correctly");
        }
      }

      /** check parallel_for_work_group(kernel, range, range, lambda) function
      */
      {
        int result = 0;

        cl::sycl::program program(queue.get_context());
        cl::sycl::kernel preBuiltKernel =
            program
                .get_kernel<parallel_for_work_group_1range_lambda_prebuilt>();

        {
          cl::sycl::buffer<int, 1> buffer(&result, cl::sycl::range<1>(1));

          queue.submit([&](cl::sycl::handler& handler) {
            cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                               cl::sycl::access::target::global_buffer>
                accessor(buffer, handler);

            cl::sycl::range<1> globalRange(2);
            cl::sycl::range<1> localRange(2);

            handler.parallel_for_work_group<
                parallel_for_work_group_1range_lambda_prebuilt>(
                globalRange, localRange, [=](cl::sycl::group<1> group) {
                  parallel_for_work_item(
                      group, [&](cl::sycl::item<1> item) { accessor[0] = 10; });

                  cl::sycl::range<1> subRange(1);

                  parallel_for_work_item(
                      preBuiltKernel, group, subRange,
                      [&](cl::sycl::item<1> item) { accessor[0] = 10; });
                });
          });
        }

        if (result != 10) {
          FAIL(log,
               "parallel_for_work_group(range, range, lambda) failed to "
               "execute correctly");
        }
      }

      /** check parallel_for_work_group(kernel, range, range, functor) function
      */
      {
        int result = 0;

        cl::sycl::program program(queue.get_context());
        cl::sycl::kernel preBuiltKernel =
            program.get_kernel<parallel_for_work_group_1range_functor>();

        {
          cl::sycl::buffer<int, 1> buffer(&result, cl::sycl::range<1>(1));

          queue.submit([&](cl::sycl::handler& handler) {
            cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                               cl::sycl::access::target::global_buffer>
                accessor(buffer, handler);

            cl::sycl::range<1> globalRange(2);
            cl::sycl::range<1> localRange(2);

            handler.parallel_for_work_group(
                preBuiltKernel, globalRange, localRange,
                parallel_for_work_group_1range_functor(accessor));
          });
        }

        if (result != 10) {
          FAIL(log,
               "parallel_for_work_group(range, range, functor) failed to "
               "execute correctly");
        }
      }
    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      FAIL(log, "a sycl exception was caught");
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace invoke_apis__ */

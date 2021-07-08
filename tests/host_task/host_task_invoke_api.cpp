/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provide verification for invoking a native C++ callable.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME host_task_invoke_api

namespace TEST_NAMESPACE {
using namespace sycl_cts;

template <typename T>
class kernel;

/** This functor object represents a command group that will be executed
 *  on a device. It performs a multiplication of every buffers` element by
 *  specified constant by calling parallel_for method.
 */
template <typename bufferT>
struct kernel_command_group {
  kernel_command_group() = delete;
  kernel_command_group(bufferT& buf, int mul)
      : m_bufRef{buf}, m_multiplier(mul) {}
  void operator()(sycl::handler& cgh) {
    int m = m_multiplier;
    auto acc_dev =
        m_bufRef.get().template get_access<sycl::access_mode::read_write>(cgh);
    cgh.parallel_for<kernel<bufferT>>(m_bufRef.get().get_range(),
                                      [=](sycl::id<1> i) { acc_dev[i] *= m; });
  }

  std::reference_wrapper<bufferT> m_bufRef;
  int m_multiplier;
};

/** This functor object represents a command group that will be executed
 *  on host. It performs an addition of a constant to every buffers` element by
 *  calling host_task method.
 */
template <typename bufferT>
struct host_task_command_group {
  host_task_command_group() = delete;
  host_task_command_group(bufferT& buf, int a) : m_bufRef{buf}, m_add(a) {}
  void operator()(sycl::handler& cgh) {
    int a = m_add;
    int container_size = m_bufRef.get().get_count();
    auto acc_host =
        m_bufRef.get()
            .template get_access<sycl::access_mode::read_write,
                                 sycl::target::host_buffer>(cgh);
    cgh.host_task([=]() {
      for (int i = 0; i < container_size; ++i) {
        acc_host[i] += a;
      }
    });
  }

  std::reference_wrapper<bufferT> m_bufRef{};
  int m_add{};
};

template <typename T>
void verify_results(sycl::vector_class<T>& data, T expected,
                    util::logger& log) {
  for (T& d : data) {
    if (d != expected) {
      FAIL(log, "Data verification failed. Expected: " +
                    std::to_string(expected) + " got: " + std::to_string(d));
    }
  }
}

class TEST_NAME : public sycl_cts::util::test_base {
  /** return information about this test
   */
  void get_info(test_base::info& out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  static constexpr int container_size{2};
  static constexpr int init_value{13};
  static constexpr int multiplier{3};
  static constexpr int add{3};

  void check_execution_kernel_host(sycl::queue& q, util::logger& log) {
    log.note("Checking device_task -> host_task execution");
    constexpr int expected{init_value * multiplier + add};
    sycl::vector_class<int> data(container_size, init_value);

    {
      sycl::buffer<int, 1> buffer(data.data(), sycl::range<1>{container_size});

      kernel_command_group<sycl::buffer<int, 1>> kernel(buffer, multiplier);
      host_task_command_group<sycl::buffer<int, 1>> host_task(buffer, add);

      q.submit(kernel);
      q.submit(host_task);
    }

    verify_results(data, expected, log);
  }

  void check_execution_host_kernel(sycl::queue& q, util::logger& log) {
    log.note("Checking host_task -> device_task execution");
    constexpr int expected = (init_value + add) * multiplier;
    sycl::vector_class<int> data(container_size, init_value);

    {
      sycl::buffer<int, 1> buffer(data.data(), sycl::range<1>{container_size});

      kernel_command_group<sycl::buffer<int, 1>> kernel(buffer, multiplier);
      host_task_command_group<sycl::buffer<int, 1>> host_task(buffer, add);

      q.submit(host_task);
      q.submit(kernel);
    }

    verify_results(data, expected, log);
  }

  void check_execution_kernel_host_kernel(sycl::queue& q, util::logger& log) {
    log.note("Checking device_task -> host_task -> device_task execution");

    constexpr int expected = (init_value * multiplier + add) * multiplier;
    sycl::vector_class<int> data(container_size, init_value);

    {
      sycl::buffer<int, 1> buffer(data.data(), sycl::range<1>{container_size});

      kernel_command_group<sycl::buffer<int, 1>> kernel_1(buffer, multiplier);
      kernel_command_group<sycl::buffer<int, 1>> kernel_2(buffer, multiplier);
      host_task_command_group<sycl::buffer<int, 1>> host_task(buffer, add);

      q.submit(kernel_1);
      q.submit(host_task);
      q.submit(kernel_2);
    }

    verify_results(data, expected, log);
  }

  void check_execution_order(sycl::queue& q, util::logger& log) {
    log.note("Checking execution order");
    check_execution_host_kernel(q, log);
    check_execution_kernel_host(q, log);
    check_execution_kernel_host_kernel(q, log);
  }

  void check_data_update(sycl::queue& q, util::logger& log) {
    constexpr int multiplier{10};
    constexpr int expected{init_value * multiplier};
    sycl::vector_class<int> data(container_size, init_value);
    {
      sycl::buffer<int, 1> buffer(data.data(), sycl::range<1>{container_size});

      q.submit([&](sycl::handler& cgh) {
        auto acc_host{
            buffer.get_access<sycl::access_mode::read,
                              sycl::target::host_buffer>(cgh)};
        auto acc_dev = buffer.get_access<sycl::access_mode::write>(cgh);
        cgh.host_task([=]() {
          for (int i = 0; i < container_size; ++i) {
            acc_dev[i] = acc_host[i] * multiplier;
          }
        });
      });

      {
        auto acc_host = buffer.get_access<sycl::access_mode::read>();
        for (int i = 0; i < container_size; ++i) {
          if (acc_host[i] != expected) {
            auto errorMessage = "Data verification failed. Expected: " +
                                std::to_string(expected) +
                                " got: " + std::to_string(acc_host[i]);
            FAIL(log, errorMessage);
          }
        }
      }
    }
  }

  void check_two_host_tasks_in_different_contexts(sycl::queue& q1,
                                                  sycl::queue& q2,
                                                  util::logger& log) {
    log.note("Checking execution of host_task in different contexts");
    constexpr int add_1{3};
    constexpr int add_2{4};
    constexpr int expected{init_value + (add_1 + add_2) * 2};
    sycl::vector_class<int> data(container_size, init_value);

    {
      sycl::buffer<int, 1> buffer(data.data(), sycl::range<1>{container_size});

      host_task_command_group<sycl::buffer<int, 1>> host_task_1(buffer, add_1);
      host_task_command_group<sycl::buffer<int, 1>> host_task_2(buffer, add_2);

      q1.submit(host_task_1);
      q2.submit(host_task_2);
      q2.submit(host_task_1);
      q1.submit(host_task_2);
    }

    verify_results(data, expected, log);
  }

  void check_host_task_and_two_kernels_in_different_contexts(
      sycl::queue& q1, sycl::queue& q2, util::logger& log) {
    log.note(
        "Checking execution of host_task and kernel in different contexts");
    constexpr int add{3};
    constexpr int expected{(init_value * multiplier + add) * multiplier + add};
    sycl::vector_class<int> data(container_size, init_value);

    {
      sycl::buffer<int, 1> buffer(data.data(), sycl::range<1>{container_size});

      kernel_command_group<sycl::buffer<int, 1>> kernel(buffer, multiplier);
      host_task_command_group<sycl::buffer<int, 1>> host_task(buffer, add);

      q1.submit(kernel);
      q2.submit(host_task);
      q2.submit(kernel);
      q1.submit(host_task);
    }

    verify_results(data, expected, log);
  }

  void check_different_contexts(sycl::queue& q1, util::logger& log) {
    sycl::queue q2{util::get_cts_object::queue()};
    check_two_host_tasks_in_different_contexts(q1, q2, log);
    check_host_task_and_two_kernels_in_different_contexts(q1, q2, log);
  }

  /** execute this test
   */
  void run(util::logger& log) override {
    try {
      sycl::queue q{util::get_cts_object::queue()};
      check_execution_order(q, log);
      check_different_contexts(q, log);
      check_data_update(q, log);
    } catch (const sycl::exception& e) {
      log_exception(log, e);
      FAIL(log, "An unexpected SYCL exception was caught");
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;
}  // namespace TEST_NAMESPACE

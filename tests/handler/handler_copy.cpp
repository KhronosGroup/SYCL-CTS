/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME handler_copy

namespace TEST_NAMESPACE {
using namespace sycl_cts;

using mode_t = cl::sycl::access::mode;
using target_t = cl::sycl::access::target;

struct simple_struct {
  simple_struct(int va = 0, float vb = 0) : a(va), b(vb) {}
  int a;
  float b;
};

/**
 * @brief Helps with getting the buffer range and filling the buffer with data
 */
template <typename dataT>
struct buffer_helper {
  static cl::sycl::range<1> construct_range(size_t elemsPerDim) {
    return {elemsPerDim};
  }
  static void fill(cl::sycl::buffer<dataT, 1>& buf, const dataT& value) {
    auto r = buf.get_range();
    auto acc = buf.template get_access<mode_t::discard_write>();
    for (int i = 0; i < r[0]; ++i) {
      acc[i] = value;
    }
  }
};

/**
 * @brief Base class that stores some data that helps with checks
 */
template <typename dataT>
struct test_single_copy_function_base {
  using host_shared_ptr = cl::sycl::shared_ptr_class<dataT>;
  using buffer_t = cl::sycl::buffer<dataT, 1>;

  static constexpr size_t elemsPerDim = 42;
  static constexpr size_t offsetPerDim = 4;
  static constexpr size_t regionPerDim = 16;
  static constexpr size_t numElems = elemsPerDim;
  static constexpr size_t bufferInitValue = 17;

  buffer_t m_bufRead =
      buffer_t(buffer_helper<dataT>::construct_range(elemsPerDim));
  buffer_t m_bufWrite =
      buffer_t(buffer_helper<dataT>::construct_range(elemsPerDim));

  host_shared_ptr m_hostPtrRead =
      host_shared_ptr(new dataT[numElems], std::default_delete<dataT[]>());
  host_shared_ptr m_hostPtrWrite =
      host_shared_ptr(new dataT[numElems], std::default_delete<dataT[]>());

  void reset_data() {
    buffer_helper<dataT>::fill(this->m_bufRead,
                               static_cast<dataT>(bufferInitValue));
    buffer_helper<dataT>::fill(this->m_bufWrite, static_cast<dataT>(0));
    for (int i = 0; i < numElems; ++i) {
      m_hostPtrRead.get()[i] = static_cast<dataT>(i);
      m_hostPtrWrite.get()[i] = static_cast<dataT>(0);
    }
  }
};

/**
 * @brief Creates an accessor based on the access target
 */
template <typename dataT, mode_t mode, target_t target>
struct accessor_helper;

template <typename dataT, mode_t mode>
struct accessor_helper<dataT, mode, target_t::global_buffer> {
  static constexpr target_t target = target_t::global_buffer;
  static cl::sycl::accessor<dataT, 1, mode, target> get(
      cl::sycl::handler& cgh, cl::sycl::buffer<dataT, 1>& buf) {
    return buf.template get_access<mode, target>(cgh);
  }
};
template <typename dataT, mode_t mode>
struct accessor_helper<dataT, mode, target_t::constant_buffer> {
  static constexpr target_t target = target_t::constant_buffer;
  static cl::sycl::accessor<dataT, 1, mode, target> get(
      cl::sycl::handler& cgh, cl::sycl::buffer<dataT, 1>& buf) {
    return buf.template get_access<mode, target>(cgh);
  }
};
template <typename dataT, mode_t mode>
struct accessor_helper<dataT, mode, target_t::host_buffer> {
  static constexpr target_t target = target_t::host_buffer;
  static cl::sycl::accessor<dataT, 1, mode, target> get(
      cl::sycl::handler& cgh, cl::sycl::buffer<dataT, 1>& buf) {
    return buf.template get_access<mode>();
  }
};
template <typename dataT, mode_t mode>
struct accessor_helper<dataT, mode, target_t::local> {
  static constexpr target_t target = target_t::local;
  static cl::sycl::accessor<dataT, 1, mode, target> get(
      cl::sycl::handler& cgh, cl::sycl::buffer<dataT, 1>& buf) {
    const auto elemsPerDim = test_single_copy_function_base<dataT>::elemsPerDim;
    auto allocationSize = buffer_helper<dataT>::construct_range(elemsPerDim);
    return cl::sycl::accessor<dataT, 1, mode, target>(allocationSize, cgh);
  }
};

/**
 * @brief Checks whether copy functions that accept a read accessor work.
 */
#define TEST_SINGLE_COPY_FUNCTION_ACC_READ(dataT, readMode, target, queue, \
                                           func)                           \
  {                                                                        \
    test_single_copy_function_base<dataT> testBase;                        \
    testBase.reset_data();                                                 \
    queue.submit([&](cl::sycl::handler& cgh) {                             \
      auto accRead = accessor_helper<dataT, readMode, target>::get(        \
          cgh, testBase.m_bufRead);                                        \
      func(cgh, accRead, testBase.m_hostPtrWrite);                         \
    });                                                                    \
    queue.wait_and_throw();                                                \
  }

/**
 * @brief Checks whether copy functions that accept a write accessor work.
 */
#define TEST_SINGLE_COPY_FUNCTION_ACC_WRITE(dataT, readMode, writeMode, \
                                            target, queue, func)        \
  {                                                                     \
    test_single_copy_function_base<dataT> testBase;                     \
    testBase.reset_data();                                              \
    queue.submit([&](cl::sycl::handler& cgh) {                          \
      auto accRead = accessor_helper<dataT, readMode, target>::get(     \
          cgh, testBase.m_bufRead);                                     \
      auto accWrite = accessor_helper<dataT, writeMode, target>::get(   \
          cgh, testBase.m_bufWrite);                                    \
      func(cgh, accRead, accWrite, testBase.m_hostPtrRead,              \
           testBase.m_hostPtrWrite);                                    \
    });                                                                 \
    queue.wait_and_throw();                                             \
  }

/**
 * @brief Creates lambdas with the actual tested functionality and passes them
 *        on to be tested. This doesn't include functions that expect a write
 *        accessor.
 */
template <typename dataT, mode_t readMode, target_t target>
void test_read_acc_copy_functions(cl::sycl::queue& queue) {
  using acc_read_t = cl::sycl::accessor<dataT, 1, readMode, target>;
  using host_shared_ptr = cl::sycl::shared_ptr_class<dataT>;

  {
    // Check copy(accessor, shared_ptr_class)
    const auto func = [](cl::sycl::handler& cgh, acc_read_t accRead,
                         host_shared_ptr hostPtrWrite) {
      cgh.copy(accRead, hostPtrWrite);
    };
    TEST_SINGLE_COPY_FUNCTION_ACC_READ(dataT, readMode, target, queue, func);
  }
  {
    // Check copy(accessor, dataT*)
    const auto func = [](cl::sycl::handler& cgh, acc_read_t accRead,
                         host_shared_ptr hostPtrWrite) {
      cgh.copy(accRead, hostPtrWrite.get());
    };
    TEST_SINGLE_COPY_FUNCTION_ACC_READ(dataT, readMode, target, queue, func);
  }
  {
    // Check update_host(accessor)
    const auto func = [](cl::sycl::handler& cgh, acc_read_t accRead,
                         host_shared_ptr hostPtrWrite) {
      cgh.update_host(accRead);
    };
    TEST_SINGLE_COPY_FUNCTION_ACC_READ(dataT, readMode, target, queue, func);
  }
}

/**
 * @brief Creates lambdas with the actual tested functionality and passes them
 *        on to be tested. This includes functions that expect a write accessor.
 */
template <typename dataT, mode_t readMode, mode_t writeMode, target_t target>
void test_write_acc_copy_functions(cl::sycl::queue& queue) {
  using acc_read_t = cl::sycl::accessor<dataT, 1, readMode, target>;
  using acc_write_t = cl::sycl::accessor<dataT, 1, writeMode, target>;
  using host_shared_ptr = cl::sycl::shared_ptr_class<dataT>;

  {
    // Check copy(shared_ptr_class, accessor)
    const auto func = [](cl::sycl::handler& cgh, acc_read_t accRead,
                         acc_write_t accWrite, host_shared_ptr hostPtrRead,
                         host_shared_ptr hostPtrWrite) {
      cgh.copy(hostPtrRead, accWrite);
    };
    TEST_SINGLE_COPY_FUNCTION_ACC_WRITE(dataT, readMode, writeMode, target,
                                        queue, func);
  }
  {
    // Check copy(dataT*, accessor)
    const auto func = [](cl::sycl::handler& cgh, acc_read_t accRead,
                         acc_write_t accWrite, host_shared_ptr hostPtrRead,
                         host_shared_ptr hostPtrWrite) {
      cgh.copy(hostPtrRead.get(), accWrite);
    };
    TEST_SINGLE_COPY_FUNCTION_ACC_WRITE(dataT, readMode, writeMode, target,
                                        queue, func);
  }
  {
    // Check copy(accessor, accessor)
    const auto func = [](cl::sycl::handler& cgh, acc_read_t accRead,
                         acc_write_t accWrite, host_shared_ptr hostPtrRead,
                         host_shared_ptr hostPtrWrite) {
      cgh.copy(accRead, accWrite);
    };
    TEST_SINGLE_COPY_FUNCTION_ACC_WRITE(dataT, readMode, writeMode, target,
                                        queue, func);
  }
  {
    // Check fill(accessor, dataT)
    const auto func = [](cl::sycl::handler& cgh, acc_read_t accRead,
                         acc_write_t accWrite, host_shared_ptr hostPtrRead,
                         host_shared_ptr hostPtrWrite) {
      const auto pattern = dataT(117);
      cgh.fill(accWrite, pattern);
    };
    TEST_SINGLE_COPY_FUNCTION_ACC_WRITE(dataT, readMode, writeMode, target,
                                        queue, func);
  }
}

/**
 * @brief Tests all valid combinations of source and destination accessor access
 *        modes.
 */
template <typename dataT>
void test_all_copy_functions(cl::sycl::queue& queue) {
  {
    // target == global_buffer

    test_read_acc_copy_functions<dataT, mode_t::read, target_t::global_buffer>(
        queue);
    test_write_acc_copy_functions<dataT, mode_t::read, mode_t::write,
                                  target_t::global_buffer>(queue);
    test_write_acc_copy_functions<dataT, mode_t::read, mode_t::read_write,
                                  target_t::global_buffer>(queue);
    test_write_acc_copy_functions<dataT, mode_t::read, mode_t::discard_write,
                                  target_t::global_buffer>(queue);
    test_write_acc_copy_functions<dataT, mode_t::read,
                                  mode_t::discard_read_write,
                                  target_t::global_buffer>(queue);

    test_read_acc_copy_functions<dataT, mode_t::read_write,
                                 target_t::global_buffer>(queue);
    test_write_acc_copy_functions<dataT, mode_t::read_write, mode_t::write,
                                  target_t::global_buffer>(queue);
    test_write_acc_copy_functions<dataT, mode_t::read_write, mode_t::read_write,
                                  target_t::global_buffer>(queue);
    test_write_acc_copy_functions<dataT, mode_t::read_write,
                                  mode_t::discard_write,
                                  target_t::global_buffer>(queue);
    test_write_acc_copy_functions<dataT, mode_t::read_write,
                                  mode_t::discard_read_write,
                                  target_t::global_buffer>(queue);
  }
  {
    // target == constant_buffer
    test_read_acc_copy_functions<dataT, mode_t::read,
                                 target_t::constant_buffer>(queue);
  }
}

/** tests the API for cl::sycl::handler
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info& out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  void run(util::logger& log) override {
    try {
      auto queue = util::get_cts_object::queue();

      test_all_copy_functions<char>(queue);
      test_all_copy_functions<short>(queue);
      test_all_copy_functions<int>(queue);
      test_all_copy_functions<long>(queue);
      test_all_copy_functions<float>(queue);
      test_all_copy_functions<double>(queue);

      test_all_copy_functions<cl::sycl::char2>(queue);
      test_all_copy_functions<cl::sycl::short3>(queue);
      test_all_copy_functions<cl::sycl::int4>(queue);
      test_all_copy_functions<cl::sycl::long8>(queue);
      test_all_copy_functions<cl::sycl::float8>(queue);
      test_all_copy_functions<cl::sycl::double16>(queue);

    } catch (const cl::sycl::exception& e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */

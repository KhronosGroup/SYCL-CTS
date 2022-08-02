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

#define TEST_NAME handler_copy

namespace TEST_NAMESPACE {
using namespace sycl_cts;

using mode_t = cl::sycl::access::mode;
using target_t = cl::sycl::access::target;

/**
 * @brief Helps with getting the buffer range and filling the buffer with data
 */
template <typename dataT, int dims>
struct buffer_helper;

template <typename dataT>
struct buffer_helper<dataT, 1> {
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

template <typename dataT>
struct buffer_helper<dataT, 2> {
  static cl::sycl::range<2> construct_range(size_t elemsPerDim) {
    return {elemsPerDim, elemsPerDim};
  }
  static void fill(cl::sycl::buffer<dataT, 2>& buf, const dataT& value) {
    auto r = buf.get_range();
    auto acc = buf.template get_access<mode_t::discard_write>();
    for (int r0 = 0; r0 < r[0]; ++r0) {
      for (int r1 = 0; r1 < r[1]; ++r1) {
        acc[r0][r1] = value;
      }
    }
  }
};

template <typename dataT>
struct buffer_helper<dataT, 3> {
  static cl::sycl::range<3> construct_range(size_t elemsPerDim) {
    return {elemsPerDim, elemsPerDim, elemsPerDim};
  }
  static void fill(cl::sycl::buffer<dataT, 3>& buf, const dataT& value) {
    auto r = buf.get_range();
    auto acc = buf.template get_access<mode_t::discard_write>();
    for (int r0 = 0; r0 < r[0]; ++r0) {
      for (int r1 = 0; r1 < r[1]; ++r1) {
        for (int r2 = 0; r2 < r[2]; ++r2) {
          acc[r0][r1][r2] = value;
        }
      }
    }
  }
};

/**
 * @brief Base class that stores some data that helps with checks
 */
template <typename dataT, int dim_src, int dim_dst>
struct test_single_copy_function_base {
  using host_shared_ptr = cl::sycl::shared_ptr_class<dataT>;
  using buffer_src_t = cl::sycl::buffer<dataT, dim_src>;
  using buffer_dst_t = cl::sycl::buffer<dataT, dim_dst>;

  static constexpr int get_elems(int dim) {
    return (dim == 1) ? 64 : (dim == 2) ? 8 : 4;
  }

  static constexpr size_t numElems = 64;
  static constexpr size_t elemsSrcPerDim = get_elems(dim_src);
  static constexpr size_t elemsDstPerDim = get_elems(dim_dst);
  static constexpr size_t bufferInitValue = 17;

  buffer_src_t m_bufRead = buffer_src_t(
      buffer_helper<dataT, dim_src>::construct_range(elemsSrcPerDim));
  buffer_dst_t m_bufWrite = buffer_dst_t(
      buffer_helper<dataT, dim_dst>::construct_range(elemsDstPerDim));

  host_shared_ptr m_hostPtrRead =
      host_shared_ptr(new dataT[numElems], std::default_delete<dataT[]>());
  host_shared_ptr m_hostPtrWrite =
      host_shared_ptr(new dataT[numElems], std::default_delete<dataT[]>());

  void reset_data() {
    buffer_helper<dataT, dim_src>::fill(this->m_bufRead,
                                        static_cast<dataT>(bufferInitValue));
    buffer_helper<dataT, dim_dst>::fill(this->m_bufWrite,
                                        static_cast<dataT>(0));
    for (int i = 0; i < numElems; ++i) {
      m_hostPtrRead.get()[i] = static_cast<dataT>(i);
      m_hostPtrWrite.get()[i] = static_cast<dataT>(0);
    }
  }
};

/**
 * @brief Creates an accessor based on the access target
 */
template <typename dataT, int dims, mode_t mode, target_t target>
struct accessor_helper;

template <typename dataT, int dims, mode_t mode>
struct accessor_helper<dataT, dims, mode, target_t::global_buffer> {
  static constexpr target_t target = target_t::global_buffer;
  static cl::sycl::accessor<dataT, dims, mode, target> get(
      cl::sycl::handler& cgh, cl::sycl::buffer<dataT, dims>& buf) {
    return buf.template get_access<mode, target>(cgh);
  }
};
template <typename dataT, int dims, mode_t mode>
struct accessor_helper<dataT, dims, mode, target_t::constant_buffer> {
  static constexpr target_t target = target_t::constant_buffer;
  static cl::sycl::accessor<dataT, dims, mode, target> get(
      cl::sycl::handler& cgh, cl::sycl::buffer<dataT, dims>& buf) {
    return buf.template get_access<mode, target>(cgh);
  }
};
template <typename dataT, int dims, mode_t mode>
struct accessor_helper<dataT, dims, mode, target_t::host_buffer> {
  static constexpr target_t target = target_t::host_buffer;
  static cl::sycl::accessor<dataT, dims, mode, target> get(
      cl::sycl::handler& cgh, cl::sycl::buffer<dataT, dims>& buf) {
    return buf.template get_access<mode>();
  }
};
template <typename dataT, int dims, mode_t mode>
struct accessor_helper<dataT, dims, mode, target_t::local> {
  static constexpr target_t target = target_t::local;
  static cl::sycl::accessor<dataT, dims, mode, target> get(
      cl::sycl::handler& cgh, cl::sycl::buffer<dataT, dims>& buf) {
    const auto elemsPerDim =
        test_single_copy_function_base<dataT, dims, dims>::numElems / dims;
    auto allocationSize =
        buffer_helper<dataT, dims>::construct_range(elemsPerDim);
    return cl::sycl::accessor<dataT, dims, mode, target>(allocationSize, cgh);
  }
};

/**
 * @brief Checks whether copy functions that accept a read accessor work.
 */
#define TEST_SINGLE_COPY_FUNCTION_ACC_READ(dataT, dims, readMode, target, \
                                           queue, func)                   \
  {                                                                       \
    test_single_copy_function_base<dataT, dims, dims> testBase;           \
    testBase.reset_data();                                                \
    queue.submit([&](cl::sycl::handler& cgh) {                            \
      auto accRead = accessor_helper<dataT, dims, readMode, target>::get( \
          cgh, testBase.m_bufRead);                                       \
      func(cgh, accRead, testBase.m_hostPtrWrite);                        \
    });                                                                   \
    queue.wait_and_throw();                                               \
  }

/**
 * @brief Checks whether copy functions that accept a write accessor work.
 */
#define TEST_SINGLE_COPY_FUNCTION_ACC_WRITE(dataT, dim_src, dim_dst, readMode, \
                                            writeMode, target, queue, func)    \
  {                                                                            \
    test_single_copy_function_base<dataT, dim_src, dim_dst> testBase;          \
    testBase.reset_data();                                                     \
    queue.submit([&](cl::sycl::handler& cgh) {                                 \
      auto accRead = accessor_helper<dataT, dim_src, readMode, target>::get(   \
          cgh, testBase.m_bufRead);                                            \
      auto accWrite = accessor_helper<dataT, dim_dst, writeMode, target>::get( \
          cgh, testBase.m_bufWrite);                                           \
      func(cgh, accRead, accWrite, testBase.m_hostPtrRead,                     \
           testBase.m_hostPtrWrite);                                           \
    });                                                                        \
    queue.wait_and_throw();                                                    \
  }

/**
 * @brief Creates lambdas with the actual tested functionality and passes them
 *        on to be tested. This doesn't include functions that expect a write
 *        accessor.
 */
template <typename dataT, int dim, mode_t readMode, target_t target>
void test_read_acc_copy_functions(cl::sycl::queue& queue) {
  using acc_read_t = cl::sycl::accessor<dataT, dim, readMode, target>;
  using host_shared_ptr = cl::sycl::shared_ptr_class<dataT>;

  {
    // Check copy(accessor, shared_ptr_class)
    const auto func = [](cl::sycl::handler& cgh, acc_read_t accRead,
                         host_shared_ptr hostPtrWrite) {
      cgh.copy(accRead, hostPtrWrite);
    };
    TEST_SINGLE_COPY_FUNCTION_ACC_READ(dataT, dim, readMode, target, queue,
                                       func);
  }
  {
    // Check copy(accessor, dataT*)
    const auto func = [](cl::sycl::handler& cgh, acc_read_t accRead,
                         host_shared_ptr hostPtrWrite) {
      cgh.copy(accRead, hostPtrWrite.get());
    };
    TEST_SINGLE_COPY_FUNCTION_ACC_READ(dataT, dim, readMode, target, queue,
                                       func);
  }
  {
    // Check update_host(accessor)
    const auto func = [](cl::sycl::handler& cgh, acc_read_t accRead,
                         host_shared_ptr hostPtrWrite) {
      cgh.update_host(accRead);
    };
    TEST_SINGLE_COPY_FUNCTION_ACC_READ(dataT, dim, readMode, target, queue,
                                       func);
  }
}

/**
 * @brief Creates lambdas with the actual tested functionality and passes them
 *        on to be tested. This includes functions that expect a write accessor.
 */
template <typename dataT, int dim_src, int dim_dst, mode_t readMode,
          mode_t writeMode, target_t target>
void test_write_acc_copy_functions(cl::sycl::queue& queue) {
  using acc_read_t = cl::sycl::accessor<dataT, dim_src, readMode, target>;
  using acc_write_t = cl::sycl::accessor<dataT, dim_dst, writeMode, target>;
  using host_shared_ptr = cl::sycl::shared_ptr_class<dataT>;

  {
    // Check copy(shared_ptr_class, accessor)
    const auto func = [](cl::sycl::handler& cgh, acc_read_t accRead,
                         acc_write_t accWrite, host_shared_ptr hostPtrRead,
                         host_shared_ptr hostPtrWrite) {
      cgh.copy(hostPtrRead, accWrite);
    };
    TEST_SINGLE_COPY_FUNCTION_ACC_WRITE(dataT, dim_src, dim_dst, readMode,
                                        writeMode, target, queue, func);
  }
  {
    // Check copy(dataT*, accessor)
    const auto func = [](cl::sycl::handler& cgh, acc_read_t accRead,
                         acc_write_t accWrite, host_shared_ptr hostPtrRead,
                         host_shared_ptr hostPtrWrite) {
      cgh.copy(hostPtrRead.get(), accWrite);
    };
    TEST_SINGLE_COPY_FUNCTION_ACC_WRITE(dataT, dim_src, dim_dst, readMode,
                                        writeMode, target, queue, func);
  }
  {
    // Check copy(accessor, accessor)
    const auto func = [](cl::sycl::handler& cgh, acc_read_t accRead,
                         acc_write_t accWrite, host_shared_ptr hostPtrRead,
                         host_shared_ptr hostPtrWrite) {
      cgh.copy(accRead, accWrite);
    };
    TEST_SINGLE_COPY_FUNCTION_ACC_WRITE(dataT, dim_src, dim_dst, readMode,
                                        writeMode, target, queue, func);
  }
  {
    // Check fill(accessor, dataT)
    const auto func = [](cl::sycl::handler& cgh, acc_read_t accRead,
                         acc_write_t accWrite, host_shared_ptr hostPtrRead,
                         host_shared_ptr hostPtrWrite) {
      const auto pattern = dataT(117);
      cgh.fill(accWrite, pattern);
    };
    TEST_SINGLE_COPY_FUNCTION_ACC_WRITE(dataT, dim_src, dim_dst, readMode,
                                        writeMode, target, queue, func);
  }
}

/**
 * @brief Tests all valid combinations of source and destination accessor access
 *        modes.
 */
template <typename dataT, int dim_src, int dim_dst>
void test_all_copy_functions(cl::sycl::queue& queue) {
  {
    // target == global_buffer

    test_read_acc_copy_functions<dataT, dim_src, mode_t::read,
                                 target_t::global_buffer>(queue);
    test_write_acc_copy_functions<dataT, dim_src, dim_dst, mode_t::read,
                                  mode_t::write, target_t::global_buffer>(
        queue);
    test_write_acc_copy_functions<dataT, dim_src, dim_dst, mode_t::read,
                                  mode_t::read_write, target_t::global_buffer>(
        queue);
    test_write_acc_copy_functions<dataT, dim_src, dim_dst, mode_t::read,
                                  mode_t::discard_write,
                                  target_t::global_buffer>(queue);
    test_write_acc_copy_functions<dataT, dim_src, dim_dst, mode_t::read,
                                  mode_t::discard_read_write,
                                  target_t::global_buffer>(queue);

    test_read_acc_copy_functions<dataT, dim_src, mode_t::read_write,
                                 target_t::global_buffer>(queue);
    test_write_acc_copy_functions<dataT, dim_src, dim_dst, mode_t::read_write,
                                  mode_t::write, target_t::global_buffer>(
        queue);
    test_write_acc_copy_functions<dataT, dim_src, dim_dst, mode_t::read_write,
                                  mode_t::read_write, target_t::global_buffer>(
        queue);
    test_write_acc_copy_functions<dataT, dim_src, dim_dst, mode_t::read_write,
                                  mode_t::discard_write,
                                  target_t::global_buffer>(queue);
    test_write_acc_copy_functions<dataT, dim_src, dim_dst, mode_t::read_write,
                                  mode_t::discard_read_write,
                                  target_t::global_buffer>(queue);
  }
  {
    // target == constant_buffer
    test_read_acc_copy_functions<dataT, dim_src, mode_t::read,
                                 target_t::constant_buffer>(queue);
  }
}

template <int dim_src, int dim_dst>
void test_all_dimensions(cl::sycl::queue& queue) {
  test_all_copy_functions<char, dim_src, dim_dst>(queue);
  test_all_copy_functions<short, dim_src, dim_dst>(queue);
  test_all_copy_functions<int, dim_src, dim_dst>(queue);
  test_all_copy_functions<long, dim_src, dim_dst>(queue);
  test_all_copy_functions<float, dim_src, dim_dst>(queue);
  test_all_copy_functions<double, dim_src, dim_dst>(queue);

  test_all_copy_functions<cl::sycl::char2, dim_src, dim_dst>(queue);
  test_all_copy_functions<cl::sycl::short3, dim_src, dim_dst>(queue);
  test_all_copy_functions<cl::sycl::int4, dim_src, dim_dst>(queue);
  test_all_copy_functions<cl::sycl::long8, dim_src, dim_dst>(queue);
  test_all_copy_functions<cl::sycl::float8, dim_src, dim_dst>(queue);
  test_all_copy_functions<cl::sycl::double16, dim_src, dim_dst>(queue);
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

      test_all_dimensions<1, 1>(queue);
      test_all_dimensions<1, 2>(queue);
      test_all_dimensions<1, 3>(queue);
      test_all_dimensions<2, 1>(queue);
      test_all_dimensions<2, 2>(queue);
      test_all_dimensions<2, 3>(queue);
      test_all_dimensions<3, 1>(queue);
      test_all_dimensions<3, 2>(queue);
      test_all_dimensions<3, 3>(queue);

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

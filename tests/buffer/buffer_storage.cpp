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

#define TEST_NAME buffer_storage

namespace buffer_storage__ {
using namespace sycl_cts;

template <typename T>
class custom_alloc {
 public:
  typedef T value_type;
  typedef T *pointer;
  typedef size_t size_type;

  T *allocate(size_t n) {
    T *mem = static_cast<T *>(malloc(sizeof(T) * n));
    if (mem != nullptr)
      return mem;
    else
      throw std::bad_alloc();
  }

  void deallocate(T *p, size_t n) {
    free(p);
    return;
  }
};

template <typename alloc, typename T, int size, int dims>
class buffer_storage_test {
 public:
  void operator()(util::logger &log, cl::sycl::range<dims> r) {
    std::unique_ptr<T[]> data(new T[size]);
    std::shared_ptr<T> data_shrd(new T[size], [](T *data) { delete[] data; });
    std::vector<T> data_vector;
    data_vector.reserve(size);

    cl::sycl::mutex_class m;

    std::fill(data.get(), (data.get() + size), 0);
    std::fill(data_shrd.get(), (data_shrd.get() + size), 0);

    {
      cl::sycl::buffer<T, dims, custom_alloc<T>> buf(data.get(), r);
      cl::sycl::buffer<T, dims, custom_alloc<T>> buf_shrd(data_shrd, r);
    }
    {
      cl::sycl::buffer<T, dims, custom_alloc<T>> buf_shrd(
          data_shrd, r,
          cl::sycl::property_list{cl::sycl::property::buffer::use_mutex(m)});
      m.lock();
      std::fill(data_shrd.get(), (data_shrd.get() + size), 0xFF);
      m.unlock();
      std::weak_ptr<T> data_final;
      buf_shrd.set_final_data(data_final);
      buf_shrd.set_write_back(true);
    }
    {
      cl::sycl::buffer<T, dims, custom_alloc<T>> buf_shrd(
          data_shrd, r,
          cl::sycl::property_list{cl::sycl::property::buffer::use_mutex(m)});
      m.lock();
      std::fill(data_shrd.get(), (data_shrd.get() + size), 0xFF);
      m.unlock();
      T *data_final = nullptr;
      buf_shrd.set_final_data(data_final);
      buf_shrd.set_write_back(false);
    }
    {
      cl::sycl::buffer<T, dims, custom_alloc<T>> buf_shrd(
          data_shrd, r,
          cl::sycl::property_list{cl::sycl::property::buffer::use_mutex(m)});
      m.lock();
      std::fill(data_shrd.get(), (data_shrd.get() + size), 0xFF);
      m.unlock();
      auto data_final = data_vector.begin();
      buf_shrd.set_final_data(data_final);
      buf_shrd.set_write_back(true);
    }
  }
};

/**
 * test cl::sycl::buffer initialization
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  template <typename alloc, typename T>
  void test_buffers(util::logger &log) {
    const int size = 32;
    cl::sycl::range<1> range1d(size);
    cl::sycl::range<2> range2d(size, size);
    cl::sycl::range<3> range3d(size, size, size);

    buffer_storage_test<alloc, T, size, 1> buf1d;
    buffer_storage_test<alloc, T, size * size, 2> buf2d;
    buffer_storage_test<alloc, T, size * size * size, 3> buf3d;

    buf1d(log, range1d);
    buf2d(log, range2d);
    buf3d(log, range3d);
  }

  /** execute the test
   */
  void run(util::logger &log) override {
    try {
      test_buffers<custom_alloc<int>, int>(log);
      test_buffers<custom_alloc<float>, float>(log);
      test_buffers<custom_alloc<double>, double>(log);
    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace buffer_storage__

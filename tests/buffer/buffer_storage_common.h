/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
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

#ifndef __SYCLCTS_TESTS_BUFFER_STORAGE_COMMON_H
#define __SYCLCTS_TESTS_BUFFER_STORAGE_COMMON_H

#include "../common/common.h"

#include <memory>

namespace buffer_storage_common {
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
  void operator()(util::logger &log, sycl::range<dims> r) {

    // Case 1 - Raw pointer
    auto data_final1 = std::make_unique<T[]>(size);

    // Case 2 - Null pointer
    T *data_final2 = nullptr;

    // Case 3 - Weak pointer
    std::shared_ptr<T[]> data_shared_ptr(new T[size]);
    std::weak_ptr<T[]> data_final3 = data_shared_ptr;

    // Case 4 - Shared pointer
    std::shared_ptr<T[]> data_final4(new T[size]);

    // Case 5 - Vector data
    std::vector<T> data_vector;
    data_vector.reserve(size);
    auto data_final5 = data_vector.begin();

    check_write_back(log, r, data_final1.get());
    check_write_back(log, r, data_final2, true /*is_nullptr*/);
    check_write_back(log, r, data_final3);
    check_write_back(log, r, data_final4);
    check_write_back(log, r, data_final5);
  }

private:
  template <typename C> void use_buffer(C final_data, sycl::range<dims> r) {
    std::shared_ptr<T[]> data_shrd(new T[size]);

    std::mutex m;

    std::fill(data_shrd.get(), (data_shrd.get() + size), 0);
    {
      sycl::buffer<T, dims, custom_alloc<T>> buf_shrd(
          data_shrd, r,
          sycl::property_list{sycl::property::buffer::use_mutex(m)});
      m.lock();
      std::fill(data_shrd.get(), (data_shrd.get() + size), 0xFF);
      m.unlock();
      buf_shrd.set_final_data(final_data);
      buf_shrd.set_write_back(true);
    }
  }

  template <template <typename T1> class C>
  void check_write_back(util::logger &log, sycl::range<dims> r,
                        C<T[]> final_data) {
    use_buffer(final_data, r);

    std::shared_ptr<T[]> ptr_shrd(final_data);
    T *ptr = ptr_shrd.get();
    for (size_t i = 0; i < size; ++i) {
      check_equal_values(ptr[i], (T)0xFF);
    }
  }

  template <typename C>
  void check_write_back(util::logger &log, sycl::range<dims> r,
                        C final_data, bool is_nullptr = false) {
    use_buffer(final_data, r);

    if (!is_nullptr) {
      for (size_t i = 0; i < size; ++i) {
        check_equal_values(final_data[i], (T)0xFF);
      }
    }
  }
};

template <typename T> class check_buffer_storage_for_type {
  template <typename alloc> void check_with_alloc(util::logger &log) {
    const int size = 32;
    sycl::range<1> range1d(size);
    sycl::range<2> range2d(size, size);
    sycl::range<3> range3d(size, size, size);

    buffer_storage_test<alloc, T, size, 1> buf1d;
    buffer_storage_test<alloc, T, size * size, 2> buf2d;
    buffer_storage_test<alloc, T, size * size * size, 3> buf3d;

    buf1d(log, range1d);
    buf2d(log, range2d);
    buf3d(log, range3d);
  }

public:
  void operator()(util::logger &log, const std::string &typeName) {
    log.note("testing: " + typeName);
    check_with_alloc<custom_alloc<T>>(log);
    check_with_alloc<sycl::buffer_allocator<T>>(log);
  }
};

} // namespace
#endif // __SYCLCTS_TESTS_BUFFER_STORAGE_COMMON_H

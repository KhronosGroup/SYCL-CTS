/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_BUFFER_STORAGE_COMMON_H
#define __SYCLCTS_TESTS_BUFFER_STORAGE_COMMON_H

#include "../common/common.h"

namespace {
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

template <typename T> class check_buffer_storage_for_type {
  template <typename alloc> void check_with_alloc(util::logger &log) {
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

public:
  void operator()(util::logger &log, const std::string &typeName) {
    log.note("testing: " + typeName);
    check_with_alloc<custom_alloc<T>>(log);
    check_with_alloc<cl::sycl::buffer_allocator>(log);
  }
};

} // namespace
#endif // __SYCLCTS_TESTS_BUFFER_STORAGE_COMMON_H
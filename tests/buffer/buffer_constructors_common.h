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

#ifndef __SYCLCTS_TESTS_BUFFER_CONSTRUCTORS_COMMON_H
#define __SYCLCTS_TESTS_BUFFER_CONSTRUCTORS_COMMON_H

#include "../common/common.h"
#include "../common/once_per_unit.h"

namespace buffer_constructors_common {
using namespace sycl_cts;

template <typename Acc>
struct iota_kernel {
  Acc acc1, acc2;
  void operator()(sycl::id<1> id) const {
    acc1[id] = static_cast<typename Acc::value_type>(id[0]);
    acc2[id] = static_cast<typename Acc::value_type>(id[0]);
  }
};

inline std::stringstream alloc_log;

template <typename T>
class logging_alloc {
 public:
  using value_type = T;
  using pointer = T *;
  using size_type = size_t;

  T *allocate(size_t n) {
    alloc_log << "Allocating " << n << " bytes of storage.\n";
    T *mem = static_cast<T *>(malloc(sizeof(T) * n));
    if (!mem) {
      alloc_log << "Failed!\n";
      throw std::bad_alloc();
    }
    return mem;
  }

  void deallocate(T *p, size_t n) {
    alloc_log << "Deallocating " << n << " bytes of storage at " << p << "\n";
    free(p);
  }
};

template <typename T, int size, int dims>
class BufferInteropNoEvent;

template <typename T, int dims, typename allocT> class BufferCheck;

template <typename T, int dims, typename allocT>
bool check_data(sycl::buffer<T, dims, allocT> buf,
                sycl::range<dims> r) {
  auto q = once_per_unit::get_queue();
  int error = 0;
  {
    sycl::buffer<int, 1> err_buf(&error, sycl::range<1>(1));
    q.submit([&](sycl::handler &cgh) {

      auto acc = buf.template get_access<sycl::access_mode::read>(cgh);
      auto err_acc =
          err_buf.template get_access<sycl::access_mode::read_write>(cgh);
      cgh.parallel_for<BufferCheck<T, dims, allocT>>(
          r, [=](sycl::id<dims> idx) {
            if (!check_equal_values(acc[idx], T {0})) {
              err_acc[0] = 1;
            }
          });
    });
  }
  return error == 0;
}

template <typename T, int dims, typename allocT>
bool check_buffer_constructor(sycl::buffer<T, dims, allocT> buf,
                              sycl::range<dims> r,
                              bool data_verify = false) {
  bool res = buf.get_range() == r;
#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
  if (data_verify) {
    res &= check_data(buf, r);
  }
#endif // SYCL_CTS_ENABLE_FULL_CONFORMANCE
  return res;
}

template <typename T, int size, int dims>
class kernel_buffer_ctor;

template <typename T, int size, int dims>
class kernel_buffer_ctor_allocator;

template <typename T, int size, int dims>
class buffer_ctors {
 public:
  void operator()(sycl::range<dims> &r, sycl::id<dims> &i,
                  const sycl::property_list &propList, util::logger &log) {
    {
      INFO("Check range constructor");
      sycl::buffer<T, dims> buf(r, propList);
      sycl::buffer<T, dims> buf1(r);
      CHECK(check_buffer_constructor(buf, r));
      CHECK(check_buffer_constructor(buf1, r));
    }

    {
      INFO("check (data pointer, range) constructor");
      T data[size];
      std::fill(data, data + size, 0);
      sycl::buffer<T, dims> buf(data, r, propList);
      sycl::buffer<T, dims> buf1(data, r);
      constexpr bool data_verify = true;
      CHECK(check_buffer_constructor(buf, r, data_verify));
      CHECK(check_buffer_constructor(buf1, r, data_verify));
    }

    {
      INFO("check (const data pointer, range) constructor");
      const T data[size] = {static_cast<T>(0)};
      sycl::buffer<T, dims> buf(data, r, propList);
      sycl::buffer<T, dims> buf1(data, r);
      constexpr bool data_verify = true;
      CHECK(check_buffer_constructor(buf, r, data_verify));
      CHECK(check_buffer_constructor(buf1, r, data_verify));
    }

    if constexpr (dims == 1) {
      INFO("check (Container)");
      std::vector<T> cont1(size), cont2(size);
      {
        sycl::buffer<T, dims> buf1(cont1, propList);
        sycl::buffer<T, dims> buf2(cont2);
        {
          once_per_unit::get_queue()
              .submit([&](sycl::handler& cgh) {
                auto acc1 =
                    buf1.template get_access<sycl::access_mode::write>(cgh);
                auto acc2 =
                    buf2.template get_access<sycl::access_mode::write>(cgh);
                cgh.parallel_for<kernel_buffer_ctor<T, size, dims>>(
                    r, iota_kernel<decltype(acc1)>{acc1, acc2});
              })
              .wait_and_throw();
        }
      }
      std::vector<T> ref(cont1.size());
      std::iota(ref.begin(), ref.end(), 0);
      CHECK(std::equal(ref.cbegin(), ref.cend(), cont1.cbegin(),
                       value_operations::are_equal<T, T>));
      CHECK(std::equal(ref.cbegin(), ref.cend(), cont2.cbegin(),
                       value_operations::are_equal<T, T>));
    }

    if constexpr (dims == 1) {
      INFO("check (Container, allocator)");
      std::vector<T> cont1(size), cont2(size);
      {
        buffer_constructors_common::logging_alloc<T> logging_alloc;
        sycl::buffer<T, dims, decltype(logging_alloc)> buf1(
            cont1, logging_alloc, propList);
        sycl::buffer<T, dims, decltype(logging_alloc)> buf2(cont2,
                                                            logging_alloc);
        {
          once_per_unit::get_queue()
              .submit([&](sycl::handler& cgh) {
                auto acc1 =
                    buf1.template get_access<sycl::access_mode::write>(cgh);
                auto acc2 =
                    buf2.template get_access<sycl::access_mode::write>(cgh);
                cgh.parallel_for<kernel_buffer_ctor_allocator<T, size, dims>>(
                    r, iota_kernel<decltype(acc1)>{acc1, acc2});
              })
              .wait_and_throw();
        }
      }
      std::vector<T> ref(cont1.size());
      std::iota(ref.begin(), ref.end(), 0);
      CHECK(std::equal(ref.cbegin(), ref.cend(), cont1.cbegin(),
                       value_operations::are_equal<T, T>));
      CHECK(std::equal(ref.cbegin(), ref.cend(), cont2.cbegin(),
                       value_operations::are_equal<T, T>));
      CHECK(alloc_log.str().empty());
    }

    {
      INFO("check (shared pointer, range) constructor");
      std::shared_ptr<T> data(new T[size], std::default_delete<T[]>());
      std::fill(data.get(), data.get() + size, 0);
      sycl::buffer<T, dims> buf(data, r, propList);
      sycl::buffer<T, dims> buf1(data, r);
      constexpr bool data_verify = true;
      CHECK(check_buffer_constructor(buf, r, data_verify));
      CHECK(check_buffer_constructor(buf1, r, data_verify));
    }

    {
      INFO("check (shared pointer[], range) constructor");
      std::shared_ptr<T[]> data(new T[size]);
      std::fill(data.get(), data.get() + size, 0);
      sycl::buffer<T, dims> buf(data, r, propList);
      sycl::buffer<T, dims> buf1(data, r);
      constexpr bool data_verify = true;
      CHECK(check_buffer_constructor(buf, r, data_verify));
      CHECK(check_buffer_constructor(buf1, r, data_verify));
    }

    if (dims == 1) {
      INFO("Check buffer iterator constructor");
      T data[size];
      std::fill(data, (data + size), 0);
      sycl::buffer<T, 1> buf_iter(data, data + size, propList);
      sycl::buffer<T, 1> buf_iter1(data, data + size);
      sycl::range<1> r_exp(size);
      constexpr bool data_verify = true;
      CHECK(check_buffer_constructor(buf_iter, r_exp, data_verify));
      CHECK(check_buffer_constructor(buf_iter1, r_exp, data_verify));
    }

    {
      INFO("Check subBuffer (buffer, id, range) constructor");
      auto r_sub = r;
      r_sub[0] = r[0] - i[0];
      sycl::buffer<T, dims> buf(r);
      sycl::buffer<T, dims> buf_sub(buf, i, r_sub);
      CHECK(buf_sub.is_sub_buffer());
      CHECK(check_buffer_constructor(buf_sub, r_sub));
    }
    /* Check range constructor */
    {
      sycl::buffer<T, dims, std::allocator<T>> buf(r, propList);
      sycl::buffer<T, dims, std::allocator<T>> buf1(r);
    }

    {
      INFO("check (data pointer, range) constructor");
      T data[size];
      std::fill(data, (data + size), 0);
      sycl::buffer<T, dims, std::allocator<T>> buf(data, r, propList);
      sycl::buffer<T, dims, std::allocator<T>> buf1(data, r);
      constexpr bool data_verify = true;
      CHECK(check_buffer_constructor(buf, r, data_verify));
      CHECK(check_buffer_constructor(buf1, r, data_verify));
    }

    {
      INFO(
          "check (const data pointer, range) constructor with allocator "
          "template param");
      const T data[size] = {static_cast<T>(0)};
      sycl::buffer<T, dims, std::allocator<T>> buf(data, r, propList);
      sycl::buffer<T, dims, std::allocator<T>> buf1(data, r);
      constexpr bool data_verify = true;
      CHECK(check_buffer_constructor(buf, r, data_verify));
      CHECK(check_buffer_constructor(buf1, r, data_verify));
    }

    {
      INFO(
          "check (shared pointer, range) constructor with allocator template "
          "param");
      std::shared_ptr<T> data(new T[size], std::default_delete<T[]>());
      std::fill(data.get(), data.get() + size, 0);
      sycl::buffer<T, dims, std::allocator<T>> buf(data, r, propList);
      sycl::buffer<T, dims, std::allocator<T>> buf1(data, r);
      constexpr bool data_verify = true;
      CHECK(check_buffer_constructor(buf, r, data_verify));
      CHECK(check_buffer_constructor(buf1, r, data_verify));
    }

    {
      INFO(
          "check (shared pointer[], range) constructor with allocator template "
          "param");
      std::shared_ptr<T[]> data(new T[size]);
      std::fill(data.get(), data.get() + size, 0);
      sycl::buffer<T, dims, std::allocator<T>> buf(data, r, propList);
      sycl::buffer<T, dims, std::allocator<T>> buf1(data, r);
      constexpr bool data_verify = true;
      CHECK(check_buffer_constructor(buf, r, data_verify));
      CHECK(check_buffer_constructor(buf1, r, data_verify));
    }

    if (dims == 1) {
      INFO("Check buffer iterator constructor with allocator template param");
      T data[size];
      std::fill(data, (data + size), 0);
      sycl::buffer<T, 1, std::allocator<T>> buf_iter(data, data + size,
                                                         propList);
      sycl::buffer<T, 1, std::allocator<T>> buf_iter1(data, data + size);
      sycl::range<1> r_exp(size);
      constexpr bool data_verify = true;
      CHECK(check_buffer_constructor(buf_iter, r_exp, data_verify));
      CHECK(check_buffer_constructor(buf_iter1, r_exp, data_verify));
    }

    {
      INFO(
          "Check subBuffer (buffer, id, range) constructor with allocator "
          "template param");
      auto r_sub = r;
      r_sub[0] = r[0] - i[0];
      sycl::buffer<T, dims, std::allocator<T>> buf(r);
      sycl::buffer<T, dims, std::allocator<T>> buf_sub(buf, i, r_sub);
      CHECK(buf_sub.is_sub_buffer());
      CHECK(check_buffer_constructor(buf_sub, r_sub));
    }

    {
      INFO("Check (range, allocator) constructor");
      sycl::buffer_allocator<T> buf_alloc;
      sycl::buffer<T, dims> buf(r, buf_alloc, propList);
      sycl::buffer<T, dims> buf1(r, buf_alloc);
      CHECK(check_buffer_constructor(buf, r));
      CHECK(check_buffer_constructor(buf1, r));
    }

    {
      INFO("check (data pointer, range, allocator) constructor");
      sycl::buffer_allocator<T> buf_alloc;
      T data[size];
      std::fill(data, (data + size), 0);
      sycl::buffer<T, dims> buf(data, r, buf_alloc, propList);
      sycl::buffer<T, dims> buf1(data, r, buf_alloc, propList);
      constexpr bool data_verify = true;
      CHECK(check_buffer_constructor(buf, r, data_verify));
      CHECK(check_buffer_constructor(buf1, r, data_verify));
    }

    {
      INFO("check (const data pointer, range, allocator) constructor");
      sycl::buffer_allocator<T> buf_alloc;
      const T data[size] = {static_cast<T>(0)};
      sycl::buffer<T, dims> buf(data, r, buf_alloc, propList);
      sycl::buffer<T, dims> buf1(data, r, buf_alloc);
      constexpr bool data_verify = true;
      CHECK(check_buffer_constructor(buf, r, data_verify));
      CHECK(check_buffer_constructor(buf1, r, data_verify));
    }

    {
      INFO("check (shared pointer, range, allocator) constructor");
      sycl::buffer_allocator<T> buf_alloc;
      std::shared_ptr<T> data(new T[size], std::default_delete<T[]>());
      std::fill(data.get(), data.get() + size, 0);
      sycl::buffer<T, dims> buf(data, r, buf_alloc, propList);
      sycl::buffer<T, dims> buf1(data, r, buf_alloc);
      constexpr bool data_verify = true;
      CHECK(check_buffer_constructor(buf, r, data_verify));
      CHECK(check_buffer_constructor(buf1, r, data_verify));
    }

    {
      INFO("check (shared pointer[], range, allocator) constructor");
      sycl::buffer_allocator<T> buf_alloc;
      std::shared_ptr<T[]> data(new T[size]);
      std::fill(data.get(), data.get() + size, 0);
      sycl::buffer<T, dims> buf(data, r, buf_alloc, propList);
      sycl::buffer<T, dims> buf1(data, r, buf_alloc);
      constexpr bool data_verify = true;
      CHECK(check_buffer_constructor(buf, r, data_verify));
      CHECK(check_buffer_constructor(buf1, r, data_verify));
    }

    if (dims == 1) {
      INFO("Check buffer (iterator, allocator) constructor");
      sycl::buffer_allocator<T> buf_alloc;
      T data[size];
      std::fill(data, (data + size), 0);
      sycl::buffer<T, 1> buf_iter(data, data + size, buf_alloc, propList);
      sycl::buffer<T, 1> buf_iter1(data, data + size, buf_alloc);
      sycl::range<1> r_exp(size);
      constexpr bool data_verify = true;
      CHECK(check_buffer_constructor(buf_iter, r_exp, data_verify));
      CHECK(check_buffer_constructor(buf_iter1, r_exp, data_verify));
    }

    {
      INFO("Check (range, std allocator) constructor");
      std::allocator<T> buf_alloc;
      sycl::buffer<T, dims, std::allocator<T>> buf(r, buf_alloc);
      CHECK(check_buffer_constructor(buf, r));
    }

    {
      INFO("check (data pointer, range, std allocator) constructor");
      std::allocator<T> buf_alloc;
      T data[size];
      std::fill(data, (data + size), 0);
      sycl::buffer<T, dims, std::allocator<T>> buf(data, r, buf_alloc);
      constexpr bool data_verify = true;
      CHECK(check_buffer_constructor(buf, r, data_verify));
    }

    {
      INFO("check (const data pointer, range, std allocator) constructor");
      std::allocator<T> buf_alloc;
      const T data[size] = {static_cast<T>(0)};
      sycl::buffer<T, dims, std::allocator<T>> buf(data, r, buf_alloc);
      constexpr bool data_verify = true;
      CHECK(check_buffer_constructor(buf, r, data_verify));
    }

    {
      INFO("check (shared pointer, range, std allocator) constructor");
      std::allocator<T> buf_alloc;
      std::shared_ptr<T> data(new T[size], std::default_delete<T[]>());
      std::fill(data.get(), data.get() + size, 0);
      sycl::buffer<T, dims, std::allocator<T>> buf(data, r, buf_alloc);
      constexpr bool data_verify = true;
      CHECK(check_buffer_constructor(buf, r, data_verify));
    }

    {
      INFO("check (shared pointer[], range, std allocator) constructor");
      std::allocator<T> buf_alloc;
      std::shared_ptr<T[]> data(new T[size]);
      std::fill(data.get(), data.get() + size, 0);
      sycl::buffer<T, dims, std::allocator<T>> buf(data, r, buf_alloc);
      constexpr bool data_verify = true;
      CHECK(check_buffer_constructor(buf, r, data_verify));
    }

    if (dims == 1) {
      INFO("Check buffer (iterator, std allocator) constructor");
      std::allocator<T> buf_alloc;
      T data[size];
      std::fill(data, (data + size), 0);
      sycl::buffer<T, 1, std::allocator<T>> buf_iter(data, data + size,
                                                         buf_alloc);
      sycl::range<1> r_exp(size);
      constexpr bool data_verify = true;
      CHECK(check_buffer_constructor(buf_iter, r_exp, data_verify));
    }

    /* Check copy assignment */
    {
      INFO("Check copy assignment");
      const sycl::property_list propertyList{
          sycl::property::buffer::use_host_ptr()};

      T data[size];
      sycl::buffer<T, dims> bufA(data, r, propertyList);
      sycl::buffer<T, dims> bufB(data, r);

      bufB = bufA;

      bool hasHostPtrProperty = bufB.template has_property<
          sycl::property::buffer::use_host_ptr>();

      CHECK(hasHostPtrProperty);
    }

    {
      INFO("Check move assignment");
      const sycl::property_list propertyList{
          sycl::property::buffer::use_host_ptr()};

      T data[size];
      sycl::buffer<T, dims> bufA(data, r, propertyList);
      sycl::buffer<T, dims> bufB(data, r);

      bufB = std::move(bufA);

      bool hasHostPtrProperty = bufB.template has_property<
          sycl::property::buffer::use_host_ptr>();

      CHECK(hasHostPtrProperty);
    }
  }
};

/** tests buffer accessors with different types
*/
template <typename T> class check_buffer_ctors_for_type {

 public:
   void operator()(util::logger &log, const std::string &typeName) {
     INFO("testing: " + typeName);

     const int size = 8;
     sycl::range<1> range1d(size);
     sycl::range<2> range2d(size, size);
     sycl::range<3> range3d(size, size, size);

     sycl::id<1> id1d(2);
     sycl::id<2> id2d(2, 0);
     sycl::id<3> id3d(2, 0, 0);

     buffer_ctors<T, size, 1> buf1d;
     buffer_ctors<T, size * size, 2> buf2d;
     buffer_ctors<T, size * size * size, 3> buf3d;

     buffer_ctors<T, size, 1> buf1d_with_properties;
     buffer_ctors<T, size * size, 2> buf2d_with_properties;
     buffer_ctors<T, size * size * size, 3> buf3d_with_properties;

     /* create property lists */

     const sycl::property_list empty_pl{};
     std::mutex mutex;
     auto context = util::get_cts_object::context();
     const sycl::property_list pl{
         sycl::property::buffer::use_mutex(mutex),
         sycl::property::buffer::context_bound(context)};

     /* test buffer constructors with empty property list */

     buf1d(range1d, id1d, empty_pl, log);
     buf2d(range2d, id2d, empty_pl, log);
     buf3d(range3d, id3d, empty_pl, log);

     /* test buffer constructors with non-empty property list */

     buf1d_with_properties(range1d, id1d, pl, log);
     buf2d_with_properties(range2d, id2d, pl, log);
     buf3d_with_properties(range3d, id3d, pl, log);
  }
};

} /* namespace */
#endif // __SYCLCTS_TESTS_BUFFER_CONSTRUCTORS_COMMON_H

/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_BUFFER_CONSTRUCTORS_COMMON_H
#define __SYCLCTS_TESTS_BUFFER_CONSTRUCTORS_COMMON_H

#include "../common/common.h"

namespace buffer_constructors_common {
using namespace sycl_cts;

template <typename T, int size, int dims>
class BufferInteropNoEvent;

template <typename T, int dims, typename allocT> class BufferCheck;

template <typename T, int dims, typename allocT>
bool check_data(sycl::buffer<T, dims, allocT> buf,
                sycl::range<dims> r) {
  auto q = util::get_cts_object::queue();
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
#ifdef SYCL_CTS_ENABLE_FULL_CONFORMANCE
  if (data_verify) {
    res &= check_data(buf, r);
  }
#endif // SYCL_CTS_ENABLE_FULL_CONFORMANCE
  return res;
}

template <typename T, int size, int dims>
class buffer_ctors {
 public:
  void operator()(sycl::range<dims> &r, sycl::id<dims> &i,
                  const sycl::property_list &propList, util::logger &log) {
    /* Check range constructor */
    {
      sycl::buffer<T, dims> buf(r, propList);
      sycl::buffer<T, dims> buf1(r);
      if (!check_buffer_constructor(buf, r) ||
          !check_buffer_constructor(buf1, r)) {
        FAIL(log, "range constructor fail.");
      }
    }

    /* check (data pointer, range) constructor*/
    {
      T data[size];
      std::fill(data, (data + size), 0);
      sycl::buffer<T, dims> buf(data, r, propList);
      sycl::buffer<T, dims> buf1(data, r);
      constexpr bool data_verify = true;
      if (!check_buffer_constructor(buf, r, data_verify) ||
          !check_buffer_constructor(buf1, r, data_verify)) {
        FAIL(log, "(data pointer, range) constructor fail.");
      }
    }

    /* check (const data pointer, range) constructor*/
    {
      const T data[size] = {static_cast<T>(0)};
      sycl::buffer<T, dims> buf(data, r, propList);
      sycl::buffer<T, dims> buf1(data, r);
      constexpr bool data_verify = true;
      if (!check_buffer_constructor(buf, r, data_verify) ||
          !check_buffer_constructor(buf1, r, data_verify)) {
        FAIL(log, "(const data pointer, range) constructor fail.");
      }
    }

    /* check (shared pointer, range) constructor*/
    {
      std::shared_ptr<T> data(new T[size]);
      std::fill(data.get(), (data.get() + size), 0);
      sycl::buffer<T, dims> buf(data, r, propList);
      sycl::buffer<T, dims> buf1(data, r);
      constexpr bool data_verify = true;
      if (!check_buffer_constructor(buf, r, data_verify) ||
          !check_buffer_constructor(buf1, r, data_verify)) {
        FAIL(log, "(shared pointer, range) constructor fail.");
      }
    }

    /* Check buffer iterator constructor */
    if (dims == 1) {
      T data[size];
      std::fill(data, (data + size), 0);
      sycl::buffer<T, 1> buf_iter(data, data + size, propList);
      sycl::buffer<T, 1> buf_iter1(data, data + size);
      sycl::range<1> r_exp(size);
      constexpr bool data_verify = true;
      if (!check_buffer_constructor(buf_iter, r_exp, data_verify) ||
          !check_buffer_constructor(buf_iter1, r_exp, data_verify)) {
        FAIL(log, "buffer iterator constructor constructor fail.");
      }
    }

    /* Check subBuffer (buffer, id, range) constructor*/
    {
      auto r_sub = r;
      r_sub[0] = r[0] - i[0];
      sycl::buffer<T, dims> buf(r);
      sycl::buffer<T, dims> buf_sub(buf, i, r_sub);
      if (!buf_sub.is_sub_buffer()) {
        FAIL(log, "buffer was not identified as a sub-buffer. (is_sub_buffer)");
      }
      if (!check_buffer_constructor(buf_sub, r_sub)) {
        FAIL(log, "(buffer, id, range) constructor constructor fail.");
      }
    }
    /* Check range constructor */
    {
      sycl::buffer<T, dims, std::allocator<T>> buf(r, propList);
      sycl::buffer<T, dims, std::allocator<T>> buf1(r);
    }

    /* check (data pointer, range) constructor*/
    {
      T data[size];
      std::fill(data, (data + size), 0);
      sycl::buffer<T, dims, std::allocator<T>> buf(data, r, propList);
      sycl::buffer<T, dims, std::allocator<T>> buf1(data, r);
      constexpr bool data_verify = true;
      if (!check_buffer_constructor(buf, r, data_verify) ||
          !check_buffer_constructor(buf1, r, data_verify)) {
        FAIL(log, "(data pointer, range) constructor fail.");
      }
    }

    /* check (const data pointer, range) constructor*/
    {
      const T data[size] = {static_cast<T>(0)};
      sycl::buffer<T, dims, std::allocator<T>> buf(data, r, propList);
      sycl::buffer<T, dims, std::allocator<T>> buf1(data, r);
      constexpr bool data_verify = true;
      if (!check_buffer_constructor(buf, r, data_verify) ||
          !check_buffer_constructor(buf1, r, data_verify)) {
        FAIL(log, "(data pointer, range) constructor fail.");
      }
    }

    /* check (shared pointer, range) constructor*/
    {
      std::shared_ptr<T> data(new T[size]);
      std::fill(data.get(), (data.get() + size), 0);
      sycl::buffer<T, dims, std::allocator<T>> buf(data, r, propList);
      sycl::buffer<T, dims, std::allocator<T>> buf1(data, r);
      constexpr bool data_verify = true;
      if (!check_buffer_constructor(buf, r, data_verify) ||
          !check_buffer_constructor(buf1, r, data_verify)) {
        FAIL(log, "(data pointer, range) constructor fail.");
      }
    }

    /* Check buffer iterator constructor */
    if (dims == 1) {
      T data[size];
      std::fill(data, (data + size), 0);
      sycl::buffer<T, 1, std::allocator<T>> buf_iter(data, data + size,
                                                         propList);
      sycl::buffer<T, 1, std::allocator<T>> buf_iter1(data, data + size);
      sycl::range<1> r_exp(size);
      constexpr bool data_verify = true;
      if (!check_buffer_constructor(buf_iter, r_exp, data_verify) ||
          !check_buffer_constructor(buf_iter1, r_exp, data_verify)) {
        FAIL(log, "buffer iterator constructor constructor fail.");
      }
    }

    /* Check subBuffer (buffer, id, range) constructor*/
    {
      auto r_sub = r;
      r_sub[0] = r[0] - i[0];
      sycl::buffer<T, dims, std::allocator<T>> buf(r);
      sycl::buffer<T, dims, std::allocator<T>> buf_sub(buf, i, r_sub);
      if (!buf_sub.is_sub_buffer()) {
        FAIL(log, "buffer was not identified as a sub-buffer. (is_sub_buffer)");
      }
      if (!check_buffer_constructor(buf_sub, r_sub)) {
        FAIL(log, "(buffer, id, range) constructor constructor fail.");
      }
    }

    /* Check (range, allocator) constructor */
    {
      sycl::buffer_allocator<T> buf_alloc;
      sycl::buffer<T, dims> buf(r, buf_alloc, propList);
      sycl::buffer<T, dims> buf1(r, buf_alloc);
      if (!check_buffer_constructor(buf, r) ||
          !check_buffer_constructor(buf1, r)) {
        FAIL(log,
             "(data pointer, range, allocator) constructor constructor fail.");
      }
    }

    /* check (data pointer, range, allocator) constructor*/
    {
      sycl::buffer_allocator<T> buf_alloc;
      T data[size];
      std::fill(data, (data + size), 0);
      sycl::buffer<T, dims> buf(data, r, buf_alloc, propList);
      sycl::buffer<T, dims> buf1(data, r, buf_alloc, propList);
      constexpr bool data_verify = true;
      if (!check_buffer_constructor(buf, r, data_verify) ||
          !check_buffer_constructor(buf1, r, data_verify)) {
        FAIL(log,
             "(data pointer, range, allocator) constructor constructor fail.");
      }
    }

    /* check (const data pointer, range, allocator) constructor*/
    {
      sycl::buffer_allocator<T> buf_alloc;
      const T data[size] = {static_cast<T>(0)};
      sycl::buffer<T, dims> buf(data, r, buf_alloc, propList);
      sycl::buffer<T, dims> buf1(data, r, buf_alloc);
      constexpr bool data_verify = true;
      if (!check_buffer_constructor(buf, r, data_verify) ||
          !check_buffer_constructor(buf1, r, data_verify)) {
        FAIL(log,
             "(const data pointer, range, allocator) constructor constructor "
             "fail.");
      }
    }

    /* check (shared pointer, range, allocator) constructor*/
    {
      sycl::buffer_allocator<T> buf_alloc;
      std::shared_ptr<T> data(new T[size]);
      std::fill(data.get(), (data.get() + size), 0);
      sycl::buffer<T, dims> buf(data, r, buf_alloc, propList);
      sycl::buffer<T, dims> buf1(data, r, buf_alloc);
      constexpr bool data_verify = true;
      if (!check_buffer_constructor(buf, r, data_verify) ||
          !check_buffer_constructor(buf1, r, data_verify)) {
        FAIL(
            log,
            "(shared pointer, range, allocator) constructor constructor fail.");
      }
    }

    /* Check buffer (iterator, allocator) constructor */
    if (dims == 1) {
      sycl::buffer_allocator<T> buf_alloc;
      T data[size];
      std::fill(data, (data + size), 0);
      sycl::buffer<T, 1> buf_iter(data, data + size, buf_alloc, propList);
      sycl::buffer<T, 1> buf_iter1(data, data + size, buf_alloc);
      sycl::range<1> r_exp(size);
      constexpr bool data_verify = true;
      if (!check_buffer_constructor(buf_iter, r_exp, data_verify) ||
          !check_buffer_constructor(buf_iter1, r_exp, data_verify)) {
        FAIL(log, "(iterator, allocator) constructor constructor fail.");
      }
    }

    /* Check (range, std allocator) constructor */
    {
      std::allocator<T> buf_alloc;
      sycl::buffer<T, dims, std::allocator<T>> buf(r, buf_alloc);
      if (!check_buffer_constructor(buf, r)) {
        FAIL(log, "(range, std allocator) constructor constructor fail.");
      }
    }

    /* check (data pointer, range, std allocator) constructor*/
    {
      std::allocator<T> buf_alloc;
      T data[size];
      std::fill(data, (data + size), 0);
      sycl::buffer<T, dims, std::allocator<T>> buf(data, r, buf_alloc);
      constexpr bool data_verify = true;
      if (!check_buffer_constructor(buf, r, data_verify)) {
        FAIL(log,
             "(data pointer, range, std allocator) constructor constructor "
             "fail.");
      }
    }

    /* check (const data pointer, range, std allocator) constructor*/
    {
      std::allocator<T> buf_alloc;
      const T data[size] = {static_cast<T>(0)};
      sycl::buffer<T, dims, std::allocator<T>> buf(data, r, buf_alloc);
      constexpr bool data_verify = true;
      if (!check_buffer_constructor(buf, r, data_verify)) {
        FAIL(log, "(const data pointer, range, std allocator) constructor "
                  "constructor fail.");
      }
    }

    /* check (shared pointer, range, std allocator) constructor*/
    {
      std::allocator<T> buf_alloc;
      std::shared_ptr<T> data(new T[size]);
      std::fill(data.get(), (data.get() + size), 0);
      sycl::buffer<T, dims, std::allocator<T>> buf(data, r, buf_alloc);
      constexpr bool data_verify = true;
      if (!check_buffer_constructor(buf, r, data_verify)) {
        FAIL(log, "(shared pointer, range, std allocator) constructor "
                  "constructor fail.");
      }
    }

    /* check (shared pointer, range, mutex, std allocator) constructor*/
    {
      std::allocator<T> buf_alloc;
      std::shared_ptr<T> data(new T[size]);
      std::fill(data.get(), (data.get() + size), 0);
      std::mutex m;
      sycl::buffer<T, dims, std::allocator<T>> buf(data, r, buf_alloc);
      constexpr bool data_verify = true;
      if (!check_buffer_constructor(buf, r, data_verify)) {
        FAIL(log, "(shared pointer, range, mutex, std allocator) constructor "
                  "constructor fail.");
      }
    }

    /* Check buffer (iterator, std allocator) constructor */
    if (dims == 1) {
      std::allocator<T> buf_alloc;
      T data[size];
      std::fill(data, (data + size), 0);
      sycl::buffer<T, 1, std::allocator<T>> buf_iter(data, data + size,
                                                         buf_alloc);
      sycl::range<1> r_exp(size);
      constexpr bool data_verify = true;
      if (!check_buffer_constructor(buf_iter, r_exp, data_verify)) {
        FAIL(log, "(iterator, std allocator) constructor constructor fail.");
      }
    }

    /* Check copy constructor */
    {
      sycl::buffer<T, dims> bufA(r);
      sycl::buffer<T, dims> bufB(bufA);
      if (bufA.get_size() != bufB.get_size()) {
        FAIL(log, "buffer was not copy constructed properly. (get_size)");
      }
      if (bufA.get_count() != bufB.get_count()) {
        FAIL(log, "buffer was not copy constructed properly. (get_count)");
      }
      if (bufA.get_range() != bufB.get_range()) {
        FAIL(log, "buffer was not copy constructed properly. (get_range)");
      }
    }

    /* Check move constructor */
    {
      sycl::buffer<T, dims> bufA(r);
      sycl::buffer<T, dims> bufB(std::move(bufA));

      if (bufB.get_range() != r) {
        FAIL(log, "buffer was not move constructed properly. (get_range)");
      }
      if (bufB.get_size() != size * sizeof(T)) {
        FAIL(log, "buffer was not move constructed properly. (get_size)");
      }
      if (bufB.get_count() != size) {
        FAIL(log, "buffer was not move constructed properly. (get_count)");
      }
    }

    /* Check copy assignment */
    {
      const sycl::property_list propertyList{
          sycl::property::buffer::use_host_ptr()};

      T data[size];
      sycl::buffer<T, dims> bufA(data, r, propertyList);
      sycl::buffer<T, dims> bufB(data, r);

      bufB = bufA;

      bool hasHostPtrProperty = bufB.template has_property<
          sycl::property::buffer::use_host_ptr>();

      if (!hasHostPtrProperty) {
        FAIL(log,
             "buffer was not copy assigned properly. "
             "(has_property<use_host_ptr>)");
      }
    }

    /* Check move assignment */
    {
      const sycl::property_list propertyList{
          sycl::property::buffer::use_host_ptr()};

      T data[size];
      sycl::buffer<T, dims> bufA(data, r, propertyList);
      sycl::buffer<T, dims> bufB(data, r);

      bufB = std::move(bufA);

      bool hasHostPtrProperty = bufB.template has_property<
          sycl::property::buffer::use_host_ptr>();

      if (!hasHostPtrProperty) {
        FAIL(log,
             "buffer was not copy assigned properly. "
             "(has_property<use_host_ptr>)");
      }
    }

    /* Check equality operator */
    {
      const auto r2 = r * 2;

      sycl::buffer<T, dims> bufA(r);
      sycl::buffer<T, dims> bufB(bufA);
      sycl::buffer<T, dims> bufC(r2);
      bufC = bufA;
      sycl::buffer<T, dims> bufD(r2);

      /* equality of copy constructed */
      if (!(bufA == bufB)) {
        FAIL(log, "buffer equality of equals failed. (copy constructor)");
      }
      /* equality of copy assigned */
      if (!(bufA == bufC)) {
        FAIL(log, "buffer equality of equals failed. (copy assignment)");
      }
      if (bufA != bufB) {
        FAIL(log,
             "buffer non-equality does not work correctly"
             "(copy constructed)");
      }
      if (bufA != bufC) {
        FAIL(log,
             "buffer non-equality does not work correctly"
             "(copy assigned)");
      }
      if (bufC == bufD) {
        FAIL(log,
             "buffer equality does not work correctly"
             "(comparing same)");
      }
      if (!(bufC != bufD)) {
        FAIL(log,
             "buffer non-equality does not work correctly"
             "(comparing same)");
      }
    }

    /* Check hashing */
    {
      sycl::buffer<T, dims> bufA(r);
      sycl::buffer<T, dims> bufB(bufA);

      std::hash<sycl::buffer<T, dims>> hasher;

      if (hasher(bufA) != hasher(bufB)) {
        FAIL(log, "buffer hashing of equals failed.");
      }
    }
  }
};

/** tests buffer accessors with different types
*/
template <typename T> class check_buffer_ctors_for_type {

 public:
   void operator()(util::logger &log, const std::string &typeName) {
     log.note("testing: " + typeName);

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

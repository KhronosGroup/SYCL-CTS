/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME buffer_constructors

namespace buffer_constructors__ {
using namespace sycl_cts;
using namespace cl::sycl;

template <typename T, int size, int dims>
class buffer_ctors {
 public:
  using fail_proxy_alias = bool (*)(sycl_cts::util::logger &log,
                                    const char *msg, int line);
  void operator()(range<dims> r, fail_proxy_alias fail_proxy,
                  util::logger &log) {
    /* Check range constructor */
    {
      cl::sycl::buffer<T, dims> buf(r);

      if (buf.get_range() != r) {
        FAIL(log, "buffer was not copy constructed properly. (get_range)");
      }
      if (buf.get_size() != size * sizeof(T)) {
        FAIL(log, "buffer was not copy constructed properly. (get_size)");
      }
      if (buf.get_count() != size) {
        FAIL(log, "buffer was not copy constructed properly. (get_count)");
      }
    }

    /* check (data pointer, range) constructor*/
    {
      T data[size];
      std::fill(data, (data + size), 0);
      cl::sycl::buffer<T, dims> buf(data, r);
    }

    /* check (const data pointer, range) constructor*/
    {
      const T data[size] = {static_cast<T>(0)};
      cl::sycl::buffer<T, dims> buf(data, r);
    }

    /* check (shared pointer, range) constructor*/
    {
      cl::sycl::shared_ptr_class<T> data(new T[size]);
      std::fill(data.get(), (data.get() + size), 0);
      cl::sycl::buffer<T, dims> buf(data, r);
    }

    /* check (shared pointer, range, mutex) constructor*/
    {
      cl::sycl::shared_ptr_class<T> data(new T[size]);
      std::fill(data.get(), (data.get() + size), 0);
      cl::sycl::mutex_class m;
      cl::sycl::buffer<T, dims> buf(data, r, &m);
    }

    /* check (unique data pointer, range) constructor*/
    {
      cl::sycl::unique_ptr_class<T> data(new T[size]);
      std::fill(data.get(), (data.get() + size), 0);
      cl::sycl::buffer<T, dims> buf(std::move(data), r);
    }

    /* Check buffer iterator constructor */
    {
      T data[size];
      std::fill(data, (data + size), 0);
      cl::sycl::buffer<T, dims> buf_iter(data, data + size);
    }

    /* Check subBuffer (buffer, range, range) constructor*/
    {
      cl::sycl::buffer<T, dims> buf(r);
      cl::sycl::range<dims> start = r / r;
      cl::sycl::range<dims> end = r - start;
      cl::sycl::buffer<T, dims> buf_sub(buf, start, end);
    }

    /* Check (range, allocator) constructor */
    {
      cl::sycl::buffer_allocator<T> buf_alloc;
      cl::sycl::buffer<T, dims> buf(r, buf_alloc);

      if (buf.get_range() != r) {
        FAIL(log, "buffer was not copy constructed properly. (get_range)");
      }
      if (buf.get_size() != size * sizeof(T)) {
        FAIL(log, "buffer was not copy constructed properly. (get_size)");
      }
      if (buf.get_count() != size) {
        FAIL(log, "buffer was not copy constructed properly. (get_count)");
      }
    }

    /* check (data pointer, range, allocator) constructor*/
    {
      cl::sycl::buffer_allocator<T> buf_alloc;
      T data[size];
      std::fill(data, (data + size), 0);
      cl::sycl::buffer<T, dims> buf(data, r, buf_alloc);
    }

    /* check (const data pointer, range, allocator) constructor*/
    {
      cl::sycl::buffer_allocator<T> buf_alloc;
      const T data[size] = {static_cast<T>(0)};
      cl::sycl::buffer<T, dims> buf(data, r, buf_alloc);
    }

    /* check (shared pointer, range, allocator) constructor*/
    {
      cl::sycl::buffer_allocator<T> buf_alloc;
      cl::sycl::shared_ptr_class<T> data(new T[size]);
      std::fill(data.get(), (data.get() + size), 0);
      cl::sycl::buffer<T, dims> buf(data, r, buf_alloc);
    }

    /* check (shared pointer, range, mutex, allocator) constructor*/
    {
      cl::sycl::buffer_allocator<T> buf_alloc;
      cl::sycl::shared_ptr_class<T> data(new T[size]);
      std::fill(data.get(), (data.get() + size), 0);
      cl::sycl::mutex_class m;
      cl::sycl::buffer<T, dims> buf(data, r, &m, buf_alloc);
    }

    /* check (unique data pointer, range, allocator) constructor*/
    {
      cl::sycl::buffer_allocator<T> buf_alloc;
      cl::sycl::unique_ptr_class<T> data(new T[size]);
      std::fill(data.get(), (data.get() + size), 0);
      cl::sycl::buffer<T, dims> buf(std::move(data), r, buf_alloc);
    }

    /* Check subBuffer (buffer, range, range, allocator) constructor*/
    {
      cl::sycl::buffer_allocator<T> buf_alloc;
      cl::sycl::buffer<T, dims> buf(r);
      cl::sycl::range<dims> start = r / r;
      cl::sycl::range<dims> end = r - start;
      cl::sycl::buffer<T, dims> buf_sub(buf, start, end, buf_alloc);
    }

    /* Check buffer (iterator, allocator) constructor */
    {
      cl::sycl::buffer_allocator<T> buf_alloc;
      T data[size];
      std::fill(data, (data + size), 0);
      cl::sycl::buffer<T, dims> buf_iter(data, data + size, buf_alloc);
    }

    /* Check (range, std allocator) constructor */
    {
      std::allocator<T> buf_alloc;
      cl::sycl::buffer<T, dims, std::allocator<T>> buf(r, buf_alloc);

      if (buf.get_range() != r) {
        FAIL(log, "buffer was not copy constructed properly. (get_range)");
      }
      if (buf.get_size() != size * sizeof(T)) {
        FAIL(log, "buffer was not copy constructed properly. (get_size)");
      }
      if (buf.get_count() != size) {
        FAIL(log, "buffer was not copy constructed properly. (get_count)");
      }
    }

    /* check (data pointer, range, std allocator) constructor*/
    {
      std::allocator<T> buf_alloc;
      T data[size];
      std::fill(data, (data + size), 0);
      cl::sycl::buffer<T, dims, std::allocator<T>> buf(data, r, buf_alloc);
    }

    /* check (const data pointer, range, std allocator) constructor*/
    {
      std::allocator<T> buf_alloc;
      const T data[size] = {static_cast<T>(0)};
      cl::sycl::buffer<T, dims, std::allocator<T>> buf(data, r, buf_alloc);
    }

    /* check (shared pointer, range, std allocator) constructor*/
    {
      std::allocator<T> buf_alloc;
      cl::sycl::shared_ptr_class<T> data(new T[size]);
      std::fill(data.get(), (data.get() + size), 0);
      cl::sycl::buffer<T, dims, std::allocator<T>> buf(data, r, buf_alloc);
    }

    /* check (shared pointer, range, mutex, std allocator) constructor*/
    {
      std::allocator<T> buf_alloc;
      cl::sycl::shared_ptr_class<T> data(new T[size]);
      std::fill(data.get(), (data.get() + size), 0);
      cl::sycl::mutex_class m;
      cl::sycl::buffer<T, dims, std::allocator<T>> buf(data, r, &m, buf_alloc);
    }

    /* check (unique data pointer, range, std allocator) constructor*/
    {
      std::allocator<T> buf_alloc;
      cl::sycl::unique_ptr_class<T> data(new T[size]);
      std::fill(data.get(), (data.get() + size), 0);
      cl::sycl::buffer<T, dims, std::allocator<T>> buf(std::move(data), r,
                                                       buf_alloc);
    }

    /* Check subBuffer (buffer, range, range, std allocator) constructor*/
    {
      std::allocator<T> buf_alloc;
      cl::sycl::buffer<T, dims> buf(r);
      cl::sycl::range<dims> start = r / r;
      cl::sycl::range<dims> end = r - start;
      cl::sycl::buffer<T, dims, std::allocator<T>> buf_sub(buf, start, end,
                                                           buf_alloc);
    }

    /* Check buffer (iterator, std allocator) constructor */
    {
      std::allocator<T> buf_alloc;
      T data[size];
      std::fill(data, (data + size), 0);
      cl::sycl::buffer<T, dims, std::allocator<T>> buf_iter(data, data + size,
                                                            buf_alloc);
    }

    /* Check copy constructor */
    {
      cl::sycl::buffer<T, dims> bufA(r);
      cl::sycl::buffer<T, dims> bufB(bufA);
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
      cl::sycl::buffer<T, dims> bufA(r);
      cl::sycl::buffer<T, dims> bufB(std::move(bufA));

      if (bufB.get_range() != r) {
        FAIL(log, "buffer was not copy constructed properly. (get_range)");
      }
      if (bufB.get_size() != size * sizeof(T)) {
        FAIL(log, "buffer was not copy constructed properly. (get_size)");
      }
      if (bufB.get_count() != size) {
        FAIL(log, "buffer was not copy constructed properly. (get_count)");
      }
    }

    /* Check copy assignment */
    {
      cl::sycl::buffer<T, dims> bufA(r);
      cl::sycl::buffer<T, dims> bufB = bufA;
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

    /* Check move assignment */
    {
      cl::sycl::buffer<T, dims> bufA(r);
      cl::sycl::buffer<T, dims> bufB = std::move(bufA);

      if (bufB.get_range() != r) {
        FAIL(log, "buffer was not copy constructed properly. (get_range)");
      }
      if (bufB.get_size() != size * sizeof(T)) {
        FAIL(log, "buffer was not copy constructed properly. (get_size)");
      }
      if (bufB.get_count() != size) {
        FAIL(log, "buffer was not copy constructed properly. (get_count)");
      }
    }

    /* Check equality operator */
    {
      cl::sycl::buffer<T, dims> bufA(r);
      cl::sycl::buffer<T, dims> bufB(bufA);
      cl::sycl::buffer<T, dims> bufC = bufA;

      /* equality of copy constructed */
      if (!(bufA == bufB)) {
        FAIL(log, "buffer equality of equals failed. (copy constructor)");
      }
      /* equality of copy assigned */
      if (!(bufA == bufC)) {
        FAIL(log, "buffer equality of equals failed. (copy assignment)");
      }
    }

    /* Check hashing */
    {
      cl::sycl::buffer<T, dims> bufA(r);
      cl::sycl::buffer<T, dims> bufB(bufA);

      cl::sycl::hash_class<cl::sycl::buffer<T, dims>> hasher;

      if (hasher(bufA) != hasher(bufB)) {
        FAIL(log, "buffer hashing of equals failed.");
      }
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
  virtual void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  template <typename T>
  void test_buffers(util::logger &log) {
    const int size = 8;
    range<1> range1d(size);
    range<2> range2d(size, size);
    range<3> range3d(size, size, size);

    buffer_ctors<T, size, 1> buf1d;
    buffer_ctors<T, size * size, 2> buf2d;
    buffer_ctors<T, size * size * size, 3> buf3d;

    buf1d(range1d, fail_proxy, log);
    buf2d(range2d, fail_proxy, log);
    buf3d(range3d, fail_proxy, log);
  }

  /** execute the test
   */
  virtual void run(util::logger &log) override {
    try {
      test_buffers<int>(log);
      test_buffers<int8_t>(log);
      test_buffers<int16_t>(log);
      test_buffers<int32_t>(log);
      test_buffers<int64_t>(log);

      test_buffers<float>(log);
      test_buffers<double>(log);

      test_buffers<float2>(log);
      test_buffers<float3>(log);
      test_buffers<float4>(log);
      test_buffers<float8>(log);
      test_buffers<float16>(log);

      test_buffers<double2>(log);
      test_buffers<double3>(log);
      test_buffers<double4>(log);
      test_buffers<double8>(log);
      test_buffers<double16>(log);
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

} /* namespace buffer_constructors__ */

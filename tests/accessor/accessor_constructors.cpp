/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME accessor_constructors

namespace accessor_constructors__ {
using namespace sycl_cts;

template <typename T, int dims, int size, cl::sycl::access::mode mode,
          cl::sycl::access::target target>
class buffer_accessor_constructors {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> &range) {
    T data[size];
    ::memset(data, 0, sizeof(data));
    cl::sycl::buffer<T, dims> buffer(data, range);
    cl::sycl::buffer<T, dims> buffer2(data, range);

    queue.submit([&](cl::sycl::handler &handler) {
      /** check (buffer, handler) constructor
      */
      cl::sycl::accessor<T, dims, mode, target> accessor(buffer, handler);

      /** check (buffer, offset, range, handler) constructor
      */
      cl::sycl::range<dims> offset = range / 2;
      cl::sycl::range<dims> r = range / 2;
      cl::sycl::accessor<T, dims, mode, target> subAccessor(buffer, handler,
                                                            offset, r);

      /* check copy constructor
      */
      {
        cl::sycl::accessor<T, dims, mode, target> accessorA(buffer, handler);
        cl::sycl::accessor<T, dims, mode, target> accessorB(accessorA);
      }

      /* check move constructor
      */
      {
        cl::sycl::accessor<T, dims, mode, target> accessorA(buffer, handler);
        cl::sycl::accessor<T, dims, mode, target> accessorB(
            std::move(accessorA));
      }

      /* check copy assignment operator
      */
      {
        cl::sycl::accessor<T, dims, mode, target> accessorA(buffer, handler);
        cl::sycl::accessor<T, dims, mode, target> accessorB = accessorA;
      }

      /* check move assignment operator
      */
      {
        cl::sycl::accessor<T, dims, mode, target> accessorA(buffer, handler);
        cl::sycl::accessor<T, dims, mode, target> accessorB =
            std::move(accessorA);
      }

      /* check equality operator
      */
      {
        cl::sycl::accessor<T, dims, mode, target> accessorA(buffer, handler);
        cl::sycl::accessor<T, dims, mode, target> accessorB(accessorA);
        cl::sycl::accessor<T, dims, mode, target> accessorC = accessorA;
        if (!(accessorA == accessorB)) {
          FAIL(log, "accessor equality of equal failed. (copy constructor)");
        }
        if (!(accessorA == accessorC)) {
          FAIL(log, "accessor equality of equal failed. (copy assignment)");
        }
      }

      /* check hashing semantics
      */
      {
        cl::sycl::accessor<T, dims, mode, target> accessorA(buffer, handler);
        cl::sycl::accessor<T, dims, mode, target> accessorB(accessorA);

        cl::sycl::hash_class<cl::sycl::accessor<T, dims, mode, target>> hasher;

        if (hasher(accessorA) == hasher(accessorB)) {
          FAIL(log, "accessor hashing of equal failed");
        }
      }

      /** dummy kernel as no kernel is required for these checks
      */
      handler.single_task(dummy_functor());
    });
  }
};

template <typename T, int dims, int size, cl::sycl::access::mode mode>
class buffer_accessor_constructors<T, dims, size, mode,
                                   cl::sycl::access::target::host_buffer> {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> &range) {
    T data[size];
    ::memset(data, 0, sizeof(data));
    cl::sycl::buffer<T, dims> buffer(data, range);

    /** check (buffer) constructor
    */
    cl::sycl::accessor<T, dims, mode, cl::sycl::access::target::host_buffer>
        accessor(buffer);

    /** check (buffer, offset, range) constructor
    */
    cl::sycl::range<dims> offset = range / 2;
    cl::sycl::range<dims> r = range / 2;

    cl::sycl::accessor<T, dims, mode, cl::sycl::access::target::host_buffer>
        subAccessor(buffer, offset, r);
  }
};

template <typename T, int dims, int size, cl::sycl::access::mode mode>
class buffer_accessor_constructors<T, dims, size, mode,
                                   cl::sycl::access::target::local> {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> &range) {
    queue.submit([&](cl::sycl::handler &handler) {
      /** check (range, handler) constructor
      */
      cl::sycl::accessor<T, dims, mode, cl::sycl::access::target::local>
          accessor(range, handler);

      /** dummy kernel as no kernel is required for these checks
      */
      handler.single_task(dummy_functor());
    });
  }
};

template <typename T, int dims, int size, cl::sycl::access::mode mode,
          cl::sycl::access::target target>
class image_accessor_constructors {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> &range) {
    char data[size];
    ::memset(data, 0, sizeof(data));
    cl::sycl::image<dims> image(data, cl::sycl::image_channel_order::rgba,
                                cl::sycl::image_channel_type::unsigned_int8,
                                range);

    queue.submit([&](cl::sycl::handler &handler) {
      /** check (image, handler) constructor
      */
      cl::sycl::accessor<T, dims, mode, target> accessor(image, handler);

      /** dummy kernel as no kernel is required for these checks
      */
      handler.single_task(dummy_functor());
    });
  }
};

template <typename T, int dims, int size, cl::sycl::access::mode mode>
class image_accessor_constructors<T, dims, size, mode,
                                  cl::sycl::access::target::host_image> {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> &range) {
    char data[size];
    ::memset(data, 0, sizeof(data));
    cl::sycl::image<dims> image(data, cl::sycl::image_channel_order::rgba,
                                cl::sycl::image_channel_type::unsigned_int8,
                                range);

    /** check (image) constructor
    */
    cl::sycl::accessor<T, dims, mode, cl::sycl::access::target::host_image>
        accessor(image);
  }
};

template <typename T, int dims, int size, cl::sycl::access::target target>
class buffer_accessor_modes {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> &range) {
    /** check buffer accessor constructors for read
    */
    buffer_accessor_constructors<T, dims, size, cl::sycl::access::mode::read,
                                 target>
        readTests;
    readTests(log, queue, range);

    /** check buffer accessor constructors for write
    */
    buffer_accessor_constructors<T, dims, size, cl::sycl::access::mode::write,
                                 target>
        writeTests;
    writeTests(log, queue, range);

    /** check buffer accessor constructors for read_write
    */
    buffer_accessor_constructors<T, dims, size,
                                 cl::sycl::access::mode::read_write, target>
        readWriteTests;
    readWriteTests(log, queue, range);

    /** check buffer accessor constructors for discard_write
    */
    buffer_accessor_constructors<T, dims, size,
                                 cl::sycl::access::mode::discard_write, target>
        discardWriteTests;
    discardWriteTests(log, queue, range);

    /** check buffer accessor constructors for discard_read_write
    */
    buffer_accessor_constructors<
        T, dims, size, cl::sycl::access::mode::discard_read_write, target>
        discardReadWriteTests;
    discardReadWriteTests(log, queue, range);
  }
};

/** Specialization for constant_buffer
*/
template <typename T, int dims, int size>
class buffer_accessor_modes<T, dims, size,
                            cl::sycl::access::target::constant_buffer> {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> &range) {
    /** check buffer accessor constructors for read
    */
    buffer_accessor_constructors<T, dims, size, cl::sycl::access::mode::read,
                                 cl::sycl::access::target::constant_buffer>
        readTests;
    readTests(log, queue, range);
  }
};

/** Specialization for local
*/
template <typename T, int dims, int size>
class buffer_accessor_modes<T, dims, size, cl::sycl::access::target::local> {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> &range) {
    /** check buffer accessor constructors for read
    */
    buffer_accessor_constructors<T, dims, size,
                                 cl::sycl::access::mode::read_write,
                                 cl::sycl::access::target::local>
        readTests;
    readTests(log, queue, range);
  }
};

/** Specialization for host_buffer
*/
template <typename T, int dims, int size>
class buffer_accessor_modes<T, dims, size,
                            cl::sycl::access::target::host_buffer> {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> &range) {
    /** check buffer accessor constructors for read
    */
    buffer_accessor_constructors<T, dims, size, cl::sycl::access::mode::read,
                                 cl::sycl::access::target::host_buffer>
        readTests;
    readTests(log, queue, range);

    /** check buffer accessor constructors for write
    */
    buffer_accessor_constructors<T, dims, size, cl::sycl::access::mode::write,
                                 cl::sycl::access::target::host_buffer>
        writeTests;
    writeTests(log, queue, range);

    /** check buffer accessor constructors for read_write
    */
    buffer_accessor_constructors<T, dims, size,
                                 cl::sycl::access::mode::read_write,
                                 cl::sycl::access::target::host_buffer>
        readWriteTests;
    readWriteTests(log, queue, range);
  }
};

template <typename T, int dims, int size, cl::sycl::access::target target>
class image_accessor_modes {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> &range) {
    /** check image accessor constructors for read
    */
    image_accessor_constructors<T, dims, size, cl::sycl::access::mode::read,
                                target>
        readTests;
    readTests(log, queue, range);

    /** check image accessor constructors for write
    */
    image_accessor_constructors<T, dims, size, cl::sycl::access::mode::write,
                                target>
        writeTests;
    writeTests(log, queue, range);
  }
};

template <typename T, int dims, int size>
class buffer_accessor_targets {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> &range) {
    /** check buffer accessor constructors for global_buffer
    */
    buffer_accessor_modes<T, dims, size,
                          cl::sycl::access::target::global_buffer>
        globalBufferTests;
    globalBufferTests(log, queue, range);

    /** check buffer accessor constructors for constant_buffer
    */
    buffer_accessor_modes<T, dims, size,
                          cl::sycl::access::target::constant_buffer>
        constantTests;
    constantTests(log, queue, range);

    /** check buffer accessor constructors for host_buffer
    */
    buffer_accessor_modes<T, dims, size, cl::sycl::access::target::host_buffer>
        hostBufferTests;
    hostBufferTests(log, queue, range);

    /** check buffer accessor constructors for local
    */
    buffer_accessor_modes<T, dims, size, cl::sycl::access::target::local>
        localTests;
    localTests(log, queue, range);
  }
};

template <typename T, int dims, int size>
class image_accessor_targets {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> &range) {
    /** check image accessor constructors for image
    */
    image_accessor_modes<T, dims, size, cl::sycl::access::target::image>
        imageTests;
    imageTests(log, queue, range);

    /** check image accessor constructors for host_image
    */
    image_accessor_modes<T, dims, size, cl::sycl::access::target::host_image>
        hostImageTests;
    hostImageTests(log, queue, range);
  }
};

template <typename T>
class buffer_accessor_dims {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue) {
    const int size = 32;

    /** check buffer accessor constructors for 1 dimension
    */
    cl::sycl::range<1> range1d(size);
    buffer_accessor_targets<T, 1, size> acc1d;
    acc1d(log, queue, range1d);

    /** check buffer accessor constructors for 2 dimension
    */
    cl::sycl::range<2> range2d(size, size);
    buffer_accessor_targets<T, 2, (size * size)> acc2d;
    acc2d(log, queue, range2d);

    /** check buffer accessor constructors for 3 dimension
    */
    cl::sycl::range<3> range3d(size, size, size);
    buffer_accessor_targets<T, 3, (size * size * size)> acc3d;
    acc3d(log, queue, range3d);
  }
};

template <typename T>
class image_accessor_dims {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue) {
    const int count = 32;
    const int size = count * 4;

    /** check image accessor constructors for 1 dimension
    */
    cl::sycl::range<1> range1d(count);
    image_accessor_targets<T, 1, size> acc1d;
    acc1d(log, queue, range1d);

    /** check image accessor constructors for 2 dimension
    */
    cl::sycl::range<2> range2d(count / 4, 4);
    image_accessor_targets<T, 2, size> acc2d;
    acc2d(log, queue, range2d);

    /** check image accessor constructors for 3 dimension
    */
    cl::sycl::range<3> range3d(count / 8, 4, 2);
    image_accessor_targets<T, 3, size> acc3d;
    acc3d(log, queue, range3d);
  }
};

struct user_struct {
  float a;
  int b;
  char c;
};

/** tests the constructors for cl::sycl::accessor
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  virtual void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute this test
   */
  virtual void run(util::logger &log) override {
    try {
      auto queue = util::get_cts_object::queue();

      /** check buffer accessor constructors for int
      */
      buffer_accessor_dims<int>()(log, queue);

      /** check buffer accessor constructors for float
      */
      buffer_accessor_dims<float>()(log, queue);

      /** check buffer accessor constructors for double
      */
      buffer_accessor_dims<double>()(log, queue);

      /** check buffer accessor constructors for char
      */
      buffer_accessor_dims<char>()(log, queue);

      /** check buffer accessor constructors for vec
      */
      buffer_accessor_dims<cl::sycl::int2>()(log, queue);

      /** check buffer accessor constructors for user_struct
      */
      buffer_accessor_dims<user_struct>()(log, queue);

      /** check image accessor constructors for int4
      */
      image_accessor_dims<cl::sycl::int4>()(log, queue);

      /** check image accessor constructors for uint4
      */
      image_accessor_dims<cl::sycl::uint4>()(log, queue);

      /** check image accessor constructors for float4
      */
      image_accessor_dims<cl::sycl::float4>()(log, queue);

      queue.wait_and_throw();
    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace accessor_constructors__ */

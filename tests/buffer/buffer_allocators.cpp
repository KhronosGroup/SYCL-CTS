/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME buffer_allocators

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace cl::sycl;

template <typename T, int size, int dims>
class buffer_allocs {
 public:
  void operator()(util::logger &log, range<dims> r) {
    unique_ptr_class<T[]> data(new T[size]);
    std::fill(data.get(), (data.get() + size), 0);

    auto q = util::get_cts_object::queue();
    cl::sycl::buffer<T, dims, map_allocator<T>> buf(data.get(), r);

    q.submit([&](cl::sycl::handler &cgh) {
      auto acc =
          buf.template get_access<cl::sycl::access::mode::read_write>(cgh);
      cgh.parallel_for<buffer_allocs<T, size, dims>>(
          r, [=](cl::sycl::id<dims> i) { acc[i] = 0; });
    });

    q.wait_and_throw();

    for (int i = 0; i < size; ++i) {
      if (data[i] != 0) {
        FAIL(log, "map_allocator is not working");
        break;
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
    const int size = 32;
    range<1> range1d(size);
    range<2> range2d(size, size);
    range<3> range3d(size, size, size);

    buffer_allocs<T, size, 1> buf1d;
    buffer_allocs<T, size * size, 2> buf2d;
    buffer_allocs<T, size * size * size, 3> buf3d;

    buf1d(log, range1d);
    buf2d(log, range2d);
    buf3d(log, range3d);
  }

  /** execute the test
  */
  virtual void run(util::logger &log) override {
    try {
      // These allocators should exist
      cl::sycl::buffer_allocator<int> buf_a;
      cl::sycl::image_allocator img_a;

      test_buffers<int>(log);
      test_buffers<float>(log);
      test_buffers<double>(log);
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

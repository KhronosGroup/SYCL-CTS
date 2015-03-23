/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME buffer_constructors

namespace buffer_constructors__ {
using namespace sycl_cts;
using namespace cl::sycl;

template <typename T, int size, int dims>
class buffer_ctors {
 public:
  void operator()(range<dims> r) {
    /* allocate host side buffer, too large for stack */
    util::UNIQUE_PTR<T[]> data(new T[size]);
    memset(data.get(), 0, sizeof(T) * size);

    cl::sycl::buffer<T, dims> buf(r);
    cl::sycl::buffer<T, dims> buf_copy(buf);
    cl::sycl::buffer<T, dims> buf_range(data.get(), r);
    cl::sycl::range<dims> start = r / r;
    cl::sycl::range<dims> end = r - start;
    cl::sycl::buffer<T, dims> buf_sub(buf, start, end);
    cl::sycl::buffer<T, dims> buf_iter(data.get(), data.get() + size);
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

    buffer_ctors<T, size, 1> buf1d;
    buffer_ctors<T, size * size, 2> buf2d;
    buffer_ctors<T, size * size * size, 3> buf3d;

    buf1d(range1d);
    buf2d(range2d);
    buf3d(range3d);
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
      FAIL(log, "sycl exception caught");
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace buffer_constructors__ */

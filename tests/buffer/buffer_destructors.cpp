/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME buffer_destructors

namespace buffer_destructors__ {
using namespace sycl_cts;
using namespace sycl_cts::util;
using namespace cl::sycl;
using namespace cl::sycl::access;

template <typename T, int size, int dims>
class buffer_dtors {
 public:
  void operator()(util::logger &log, range<dims> r) {
    unique_ptr_class<T[]> data(new T[size]);
    std::fill(data.get(), (data.get() + size), 0);

    {
      cl::sycl::buffer<T, dims> buf(data.get(), r);
      cl::sycl::accessor<T, dims, mode::read_write, target::host_buffer> acc(
          buf);
      for (int i = 0; i < size; ++i) acc[i] = static_cast<T>(i);
    }

    for (int i = 0; i < size; ++i)
      if (data[i] != static_cast<T>(i)) {
        FAIL(log,
             "Data does not seem to have been copied on buffer "
             "destruction");
        break;
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

  template <typename T>
  void test_buffers(util::logger &log) {
    const int size = 32;
    range<1> range1d(size);
    buffer_dtors<T, size, 1> buf1d;
    buf1d(log, range1d);
  }

  /** execute the test
   */
  void run(util::logger &log) override {
    try {
      test_buffers<int>(log);
      test_buffers<float>(log);
      test_buffers<double>(log);
    } catch (const cl::sycl::exception &e) {
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

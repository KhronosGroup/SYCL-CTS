/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME item_2d

namespace test_item_2d__ {
using namespace sycl_cts;

class kernel_item_2d {
 protected:
  typedef cl::sycl::accessor<int, 2, cl::sycl::access::mode::read,
                             cl::sycl::access::target::global_buffer>
      t_readAccess;
  typedef cl::sycl::accessor<int, 2, cl::sycl::access::mode::write,
                             cl::sycl::access::target::global_buffer>
      t_writeAccess;

  t_readAccess m_x;
  t_writeAccess m_o;

 public:
  kernel_item_2d(t_readAccess in_, t_writeAccess out_) : m_x(in_), m_o(out_) {}

  void operator()(cl::sycl::item<2> item) const {
    cl::sycl::id<2> gid = item.get_id();

    size_t dim_a = item.get_id(0) + item.get_id(1);
    size_t dim_b = item[0] + item[1];

    cl::sycl::range<2> localRange = item.get_range();

    cl::sycl::id<2> offset = item.get_offset();

    /* get work item range */
    const size_t nWidth = localRange.get(0);
    const size_t nHeight = localRange.get(1);

    /* find the row major array id for this work item */
    size_t index = gid.get(1) +            /* y */
                   (gid.get(0) * nHeight); /* x */

    /* get the global linear id */
    const size_t glid = item.get_linear_id();

    /* compare against the precomputed index */
    m_o[item] = (m_x[item] == static_cast<int>(index));
  }
};

void buffer_fill(int *buf, const int nWidth, const int nHeight) {
  for (int j = 0; j < nHeight; j++) {
    for (int i = 0; i < nWidth; i++) {
      const int index = j * nWidth + i;
      buf[index] = index;
    }
  }
}

int buffer_verify(int *buf, const int nWidth, const int nHeight) {
  int nErrors = 0;
  for (int i = 0; i < nWidth * nHeight; i++) {
    nErrors += (buf[i] == 0);
  }
  return nErrors;
}

bool test_item_2d(util::logger &log) {
  const int nWidth = 8;
  const int nHeight = 8;

  const int nSize = nWidth * nHeight;

  /* allocate host buffers */
  std::unique_ptr<int> dataIn(new int[nSize]);
  std::unique_ptr<int> dataOut(new int[nSize]);

  /* clear host buffers */
  ::memset(dataIn.get(), 0, nSize * sizeof(int));
  ::memset(dataOut.get(), 0, nSize * sizeof(int));

  /*  */
  buffer_fill(dataIn.get(), nWidth, nHeight);

  try {
    cl::sycl::range<2> dataRange_i(nWidth, nHeight);
    cl::sycl::range<2> dataRange_o(nWidth, nHeight);

    cl::sycl::buffer<int, 2> bufIn(dataIn.get(), dataRange_i);
    cl::sycl::buffer<int, 2> bufOut(dataOut.get(), dataRange_o);

    auto cmdQueue = util::get_cts_object::queue();

    cmdQueue.submit([&](cl::sycl::handler &cgh) {
      auto accIn = bufIn.template get_access<cl::sycl::access::mode::read>(cgh);
      auto accOut =
          bufOut.template get_access<cl::sycl::access::mode::write>(cgh);

      kernel_item_2d kern = kernel_item_2d(accIn, accOut);

      auto r = cl::sycl::range<2>(nWidth, nHeight);
      cgh.parallel_for(r, kern);
    });

    cmdQueue.wait_and_throw();
  } catch (const cl::sycl::exception &e) {
    log_exception(log, e);
    cl::sycl::string_class errorMsg =
        "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
    FAIL(log, errorMsg.c_str());
    return false;
  }

  if (buffer_verify(dataOut.get(), nWidth, nHeight)) {
    FAIL(log, "item incorrectly mapped");
    return false;
  }

  return true;
}

/** test cl::sycl::device initialization
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  void run(util::logger &log) override { test_item_2d(log); }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace test_item_2d__ */

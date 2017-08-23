/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME item_2d

namespace test_item_2d__ {
using namespace sycl_cts;
using namespace cl::sycl;

class kernel_item_2d {
 protected:
  typedef accessor<int, 2, cl::sycl::access::mode::read,
                   cl::sycl::access::target::global_buffer>
      t_readAccess;
  typedef accessor<int, 2, cl::sycl::access::mode::write,
                   cl::sycl::access::target::global_buffer>
      t_writeAccess;

  t_readAccess m_x;
  t_writeAccess m_o;

 public:
  kernel_item_2d(t_readAccess in_, t_writeAccess out_) : m_x(in_), m_o(out_) {}

  void operator()(item<2> item) {
    id<2> gid = item.get_global();

    size_t dim_a = item.get(0);
    size_t dim_b = item[0];

    range<2> globalRange = item.get_global_range();
    range<2> localRange = item.get_local_range();

    size_t group = item.get_group(0);

    id<1> offset = item.get_offset();

    /* get work item range */
    const size_t nWidth = globalRange.get()[0];
    const size_t nHeight = globalRange.get()[1];

    /* find the array id for this work item */
    size_t index = gid.get()[0] +           /* x */
                   (gid.get()[1] * nWidth); /* y */

    /* get the global linear id */

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
  memset(dataIn.get(), 0, nSize * sizeof(int));
  memset(dataOut.get(), 0, nSize * sizeof(int));

  /*  */
  buffer_fill(dataIn.get(), nWidth, nHeight);

  try {
    range<2> dataRange_i(nWidth, nHeight);
    range<2> dataRange_o(nWidth, nHeight);

    buffer<int, 2> bufIn(dataIn.get(), dataRange_i);
    buffer<int, 2> bufOut(dataOut.get(), dataRange_o);

    cts_selector selector;
    queue cmdQueue(selector);

    cmdQueue.submit([&](handler &cgh) {
      auto accIn = bufIn.template get_access<cl::sycl::access::mode::read>(cgh);
      auto accOut =
          bufOut.template get_access<cl::sycl::access::mode::write>(cgh);

      kernel_item_2d kern = kernel_item_2d(accIn, accOut);

      auto r = nd_range<2>(range<2>(nWidth, nHeight), range<2>(1, 1));
      cgh.parallel_for(r, kern);
    });

    cmdQueue.wait_and_throw();

  } catch (cl::sycl::exception e) {
    log_exception(log, e);
    FAIL(log, "sycl exception caught");
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
  virtual void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  virtual void run(util::logger &log) override { test_item_2d(log); }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace test_item_2d__ */

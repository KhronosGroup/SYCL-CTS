/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/
#include "../common/common.h"

#define TEST_NAME item_3d

namespace test_item_3d__ {
using namespace sycl_cts;

class kernel_item_3d {
 protected:
  typedef sycl::accessor<int, 3, sycl::access_mode::read,
                             sycl::target::device>
      t_readAccess;
  typedef sycl::accessor<int, 3, sycl::access_mode::write,
                             sycl::target::device>
      t_writeAccess;

  t_readAccess m_x;
  t_writeAccess m_o;
  sycl::range<3> r_exp;
  sycl::id<3> offset_exp;

 public:
  kernel_item_3d(t_readAccess in_, t_writeAccess out_, sycl::range<3> r)
      : m_x(in_), m_o(out_), r_exp(r), offset_exp(sycl::id<3>(0, 0, 0)) {}

  void operator()(sycl::item<3> item) const {
    bool result = true;
    sycl::id<3> gid = item.get_id();

    size_t dim_a = item.get_id(0) + item.get_id(1) + item.get_id(2);
    size_t dim_b = item[0] + item[1] + item[2];
    size_t id_exp = gid.get(0) + gid.get(1) + gid.get(2);
    result &= id_exp == dim_a && id_exp == dim_b;

    sycl::range<3> localRange = item.get_range();
    result &= localRange == r_exp;

    // TODO: mark this check as testing deprecated functionality
    sycl::id<3> offset = item.get_offset();
    result &= offset == offset_exp;

    /* get work item range */
    const size_t nWidth = localRange.get(0);
    const size_t nHeight = localRange.get(1);
    const size_t nDepth = localRange.get(2);
    result &= nWidth == r_exp.get(0) && nHeight == r_exp.get(1) &&
              nDepth == r_exp.get(2);

    /* find the row major array id for this work item */
    size_t index = gid.get(2) +                     /* z */
                   (gid.get(1) * nWidth) +          /* y */
                   (gid.get(0) * nWidth * nHeight); /* x */

    /* get the global linear id */
    size_t linear_index = item.get_linear_id();

    /* compare against the precomputed index */
    result &= m_x[item] == static_cast<int>(index);
    m_o[item] = result;
  }
};

void buffer_fill(int *buf, const int nWidth, const int nHeight,
                 const int nDepth) {
  for (int k = 0; k < nDepth; k++) {
    for (int j = 0; j < nHeight; j++) {
      for (int i = 0; i < nWidth; i++) {
        const int index = (k * nWidth * nHeight) + (j * nWidth) + i;
        buf[index] = index;
      }
    }
  }
}

int buffer_verify(int *buf, const int nWidth, const int nHeight,
                  const int nDepth) {
  int nErrors = 0;
  for (int i = 0; i < nWidth * nHeight * nDepth; i++) {
    nErrors += (buf[i] == 0);
  }
  return nErrors;
}

bool test_item_3d(util::logger &log) {
  const int nWidth = 64;
  const int nHeight = 64;
  const int nDepth = 64;

  const int nSize = nWidth * nHeight * nDepth;

  /* allocate host buffers */
  std::unique_ptr<int> dataIn(new int[nSize]);
  std::unique_ptr<int> dataOut(new int[nSize]);

  /* clear host buffers */
  ::memset(dataIn.get(), 0, nSize * sizeof(int));
  ::memset(dataOut.get(), 0, nSize * sizeof(int));

  buffer_fill(dataIn.get(), nWidth, nHeight, nDepth);

  {
    sycl::range<3> dataRange(nWidth, nHeight, nDepth);

    sycl::buffer<int, 3> bufIn(dataIn.get(), dataRange);
    sycl::buffer<int, 3> bufOut(dataOut.get(), dataRange);

    auto cmdQueue = util::get_cts_object::queue();

    cmdQueue.submit([&](sycl::handler &cgh) {
      auto accIn = bufIn.template get_access<sycl::access_mode::read>(cgh);
      auto accOut =
          bufOut.template get_access<sycl::access_mode::write>(cgh);

      auto r = sycl::range<3>(dataRange);
      kernel_item_3d kern = kernel_item_3d(accIn, accOut, r);

      cgh.parallel_for(r, kern);
    });

    cmdQueue.wait_and_throw();
  }

  if (buffer_verify(dataOut.get(), nWidth, nHeight, nDepth)) {
    FAIL(log, "item incorrectly mapped");
    return false;
  }

  return true;
}

/** test sycl::device initialization
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
  void run(util::logger &log) override { test_item_3d(log); }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace test_item_2d__ */

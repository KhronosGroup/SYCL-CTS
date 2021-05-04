/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME item_1d

namespace test_item_1d__ {
using namespace sycl_cts;

class kernel_item_1d {
 protected:
  typedef cl::sycl::accessor<int, 1, cl::sycl::access::mode::read,
                             cl::sycl::access::target::global_buffer>
      t_readAccess;
  typedef cl::sycl::accessor<int, 1, cl::sycl::access::mode::write,
                             cl::sycl::access::target::global_buffer>
      t_writeAccess;

  t_readAccess m_x;
  t_writeAccess m_o;
  cl::sycl::range<1> r_exp;
  cl::sycl::id<1> offset_exp;

 public:
  kernel_item_1d(t_readAccess in_, t_writeAccess out_, cl::sycl::range<1> r)
      : m_x(in_), m_o(out_), r_exp(r), offset_exp(cl::sycl::id<1>(0)) {}

  void operator()(cl::sycl::item<1> item) const {
    bool result = true;

    cl::sycl::id<1> gid = item.get_id();

    size_t dim_a = item.get_id(0);
    size_t dim_b = item[0];
    result &= (gid.get(0) == dim_a) && (gid.get(0) == dim_b);

    cl::sycl::range<1> localRange = item.get_range();
    result &= localRange == r_exp;

    cl::sycl::id<1> offset = item.get_offset();
    (void)offset; // silent warning
    result &= offset == offset_exp;

    /* get work item range */
    const size_t nWidth = item.get_range(0);
    result &= nWidth == r_exp.get(0);

    /* find the array id for this work item */
    size_t index = gid.get(0); /* x */

    /* get the linear id */
    const size_t glid = item.get_linear_id();
    result &= m_x[int(glid)] == static_cast<int>(index);

    /* compare against the precomputed index */
    m_o[int(glid)] = result;
  }
};

void buffer_fill(int *buf, const int nWidth) {
  for (int i = 0; i < nWidth; i++) buf[i] = i;
}

int buffer_verify(int *buf, const int nWidth) {
  int nErrors = 0;
  for (int i = 0; i < nWidth; i++) nErrors += (buf[i] == 0);
  return nErrors;
}

bool test_item_1d(util::logger &log) {
  const int nWidth = 64;

  /* allocate host buffers */
  std::unique_ptr<int> dataIn(new int[nWidth]);
  std::unique_ptr<int> dataOut(new int[nWidth]);

  /* clear host buffers */
  ::memset(dataIn.get(), 0, nWidth * sizeof(int));
  ::memset(dataOut.get(), 0, nWidth * sizeof(int));

  /*  */
  buffer_fill(dataIn.get(), nWidth);

  try {
    cl::sycl::range<1> dataRange(nWidth);

    cl::sycl::buffer<int, 1> bufIn(dataIn.get(), dataRange);
    cl::sycl::buffer<int, 1> bufOut(dataOut.get(), dataRange);

    auto cmdQueue = util::get_cts_object::queue();

    cmdQueue.submit([&](cl::sycl::handler &cgh) {
      auto accIn = bufIn.template get_access<cl::sycl::access::mode::read>(cgh);
      auto accOut =
          bufOut.template get_access<cl::sycl::access::mode::write>(cgh);

      kernel_item_1d kern = kernel_item_1d(accIn, accOut, dataRange);
      cgh.parallel_for(cl::sycl::range<1>(dataRange), kern);
    });

    cmdQueue.wait_and_throw();
  } catch (const cl::sycl::exception &e) {
    log_exception(log, e);
    cl::sycl::string_class errorMsg =
        "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
    FAIL(log, errorMsg.c_str());
    return false;
  }

  /*  */
  if (buffer_verify(dataOut.get(), nWidth)) {
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
  void run(util::logger &log) override { test_item_1d(log); }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace test_item_1d__ */

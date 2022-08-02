/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2022 The Khronos Group Inc.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME item_2d

namespace test_item_2d__ {
using namespace sycl_cts;

class kernel_item_2d {
 protected:
  typedef sycl::accessor<int, 2, sycl::access_mode::read,
                             sycl::target::device>
      t_readAccess;
  typedef sycl::accessor<int, 2, sycl::access_mode::write,
                             sycl::target::device>
      t_writeAccess;

  t_readAccess m_x;
  t_writeAccess m_o;
  sycl::range<2> r_exp;
  sycl::id<2> offset_exp;

 public:
  kernel_item_2d(t_readAccess in_, t_writeAccess out_, sycl::range<2> r)
      : m_x(in_), m_o(out_), r_exp(r), offset_exp(sycl::id<2>(0, 0)) {}

  void operator()(sycl::item<2> item) const {
    bool result = true;

    sycl::id<2> gid = item.get_id();

    size_t dim_a = item.get_id(0) + item.get_id(1);
    size_t dim_b = item[0] + item[1];
    result &= (gid.get(0) + gid.get(1)) == dim_a &&
              (gid.get(0) + gid.get(1)) == dim_b;

    sycl::range<2> localRange = item.get_range();
    result &= localRange == r_exp;

    // TODO: mark this check as testing deprecated functionality
    sycl::id<2> offset = item.get_offset();
    result &= offset == offset_exp;

    /* get work item range */
    const size_t nWidth = item.get_range(0);
    const size_t nHeight = item.get_range(1);
    result &= nWidth == r_exp.get(0) && nHeight == r_exp.get(1);

    /* find the row major array id for this work item */
    size_t index = gid.get(1) +            /* y */
                   (gid.get(0) * nHeight); /* x */

    /* get the global linear id */
    const size_t glid = item.get_linear_id();

    /* compare against the precomputed index */
    result &= m_x[item] == static_cast<int>(index);
    m_o[item] = result;
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

  {
    sycl::range<2> dataRange_i(nWidth, nHeight);
    sycl::range<2> dataRange_o(nWidth, nHeight);

    sycl::buffer<int, 2> bufIn(dataIn.get(), dataRange_i);
    sycl::buffer<int, 2> bufOut(dataOut.get(), dataRange_o);

    auto cmdQueue = util::get_cts_object::queue();

    cmdQueue.submit([&](sycl::handler &cgh) {
      auto accIn = bufIn.template get_access<sycl::access_mode::read>(cgh);
      auto accOut =
          bufOut.template get_access<sycl::access_mode::write>(cgh);

      auto r = sycl::range<2>(nWidth, nHeight);
      kernel_item_2d kern = kernel_item_2d(accIn, accOut, r);

      cgh.parallel_for(r, kern);
    });

    cmdQueue.wait_and_throw();
  }

  if (buffer_verify(dataOut.get(), nWidth, nHeight)) {
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
  void run(util::logger &log) override { test_item_2d(log); }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace test_item_2d__ */

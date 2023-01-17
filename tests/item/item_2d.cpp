/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2022-2023 The Khronos Group Inc.
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

#include <functional>
#include <numeric>

#define TEST_NAME item_2d

namespace test_item_2d__ {
using namespace sycl_cts;

class kernel_item_2d {
 protected:
  using t_readAccess =
      sycl::accessor<int, 2, sycl::access_mode::read, sycl::target::device>;
  using t_writeAccess =
      sycl::accessor<int, 2, sycl::access_mode::write, sycl::target::device>;

  t_readAccess in;
  t_writeAccess out;
  t_writeAccess out_deprecated;
  sycl::range<2> r_exp;
  sycl::id<2> offset_exp;

 public:
  kernel_item_2d(t_readAccess in_, t_writeAccess out_,
                 t_writeAccess out_deprecated_, sycl::range<2> r)
      : in(in_),
        out(out_),
        out_deprecated(out_deprecated_),
        r_exp(r),
        offset_exp(sycl::id<2>(0, 0)) {}

  void operator()(sycl::item<2> item) const {
    bool all_correct = true;
    sycl::id<2> gid = item.get_id();

    size_t dim_a = item.get_id(0) + item.get_id(1);
    size_t dim_b = item[0] + item[1];
    all_correct &= (gid.get(0) + gid.get(1)) == dim_a &&
                   (gid.get(0) + gid.get(1)) == dim_b;

    sycl::range<2> localRange = item.get_range();
    all_correct &= localRange == r_exp;

#ifdef SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
    sycl::id<2> offset = item.get_offset();
    out_deprecated[item] = offset == offset_exp;
#endif

    /* get work item range */
    const size_t nWidth = item.get_range(0);
    const size_t nHeight = item.get_range(1);
    all_correct &= nWidth == r_exp.get(0) && nHeight == r_exp.get(1);

    /* find the row major array id for this work item */
    size_t index = gid.get(1) +            /* y */
                   (gid.get(0) * nHeight); /* x */

    /* get the global linear id and compare against precomputed index */
    const size_t glid = item.get_linear_id();
    all_correct &= in[item] == static_cast<int>(index);

    out[item] = all_correct;
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

void test_item_2d(util::logger &log) {
  const int nWidth = 8;
  const int nHeight = 8;

  const int nSize = nWidth * nHeight;

  /* allocate and clear host buffers */
  std::vector<int> dataIn(nSize, 0);
  std::vector<int> dataOut(nSize, 0);
  std::vector<int> dataOutDeprecated(nSize, 0);

  buffer_fill(dataIn.data(), nWidth, nHeight);

  {
    sycl::range<2> dataRange(nWidth, nHeight);

    sycl::buffer<int, 2> bufIn(dataIn.data(), dataRange);
    sycl::buffer<int, 2> bufOut(dataOut.data(), dataRange);
    sycl::buffer<int, 2> bufOutDeprecated(dataOutDeprecated.data(), dataRange);

    auto cmdQueue = util::get_cts_object::queue();

    cmdQueue.submit([&](sycl::handler &cgh) {
      auto accIn = bufIn.template get_access<sycl::access_mode::read>(cgh);
      auto accOut = bufOut.template get_access<sycl::access_mode::write>(cgh);
      auto accOutDeprecated =
          bufOutDeprecated.template get_access<sycl::access_mode::write>(cgh);

      kernel_item_2d kern =
          kernel_item_2d(accIn, accOut, accOutDeprecated, dataRange);
      cgh.parallel_for(dataRange, kern);
    });

    cmdQueue.wait_and_throw();
  }

  // check api call results
  CHECK(std::reduce(dataOut.begin(), dataOut.end(), true,
                    std::logical_and<int>{}));

#ifdef SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
  // check deprecated api call results
  CHECK(std::reduce(dataOutDeprecated.begin(), dataOutDeprecated.end(), true,
                    std::logical_and<int>{}));
#endif

  STATIC_CHECK_FALSE(std::is_default_constructible_v<sycl::item<2>>);
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

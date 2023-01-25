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

#include <algorithm>

#define TEST_NAME item_3d

namespace test_item_3d__ {
using namespace sycl_cts;

class kernel_item_3d {
 protected:
  using read_access_t =
      sycl::accessor<int, 3, sycl::access_mode::read, sycl::target::device>;
  using write_access_t =
      sycl::accessor<int, 3, sycl::access_mode::write, sycl::target::device>;

  read_access_t in;
  write_access_t out;
  write_access_t out_deprecated;
  sycl::range<3> r_exp;
  sycl::id<3> offset_exp;

 public:
  kernel_item_3d(read_access_t in_, write_access_t out_,
                 write_access_t out_deprecated_, sycl::range<3> r)
      : in(in_),
        out(out_),
        out_deprecated(out_deprecated_),
        r_exp(r),
        offset_exp(sycl::id<3>(0, 0, 0)) {}

  void operator()(sycl::item<3> item) const {
    bool all_correct = true;
    sycl::id<3> gid = item.get_id();

    for (std::size_t i = 0; i < 3; i++) {
      all_correct &= gid.get(i) == item.get_id(i);
      all_correct &= gid.get(i) == item[i];
    }

    sycl::range<3> localRange = item.get_range();
    all_correct &= localRange == r_exp;

#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
    sycl::id<3> offset = item.get_offset();
    out_deprecated[item] = offset == offset_exp;
#endif

    /* get work item range */
    const size_t nWidth = localRange.get(0);
    const size_t nHeight = localRange.get(1);
    const size_t nDepth = localRange.get(2);
    all_correct &= nWidth == r_exp.get(0) && nHeight == r_exp.get(1) &&
                   nDepth == r_exp.get(2);

    /* find the row major array id for this work item */
    size_t index = gid.get(2) +                     /* z */
                   (gid.get(1) * nWidth) +          /* y */
                   (gid.get(0) * nWidth * nHeight); /* x */

    /* get the global linear id and compare against precomputed index */
    size_t linear_index = item.get_linear_id();
    all_correct &= in[item] == static_cast<int>(index);

    out[item] = all_correct;
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

void test_item_3d(util::logger &log) {
  const int nWidth = 64;
  const int nHeight = 64;
  const int nDepth = 64;

  const int nSize = nWidth * nHeight * nDepth;

  /* allocate and clear host buffers */
  std::vector<int> dataIn(nSize);
  std::vector<int> dataOut(nSize);
  std::vector<int> dataOutDeprecated(nSize);

  buffer_fill(dataIn.data(), nWidth, nHeight, nDepth);

  {
    sycl::range<3> dataRange(nWidth, nHeight, nDepth);

    sycl::buffer<int, 3> bufIn(dataIn.data(), dataRange);
    sycl::buffer<int, 3> bufOut(dataOut.data(), dataRange);
    sycl::buffer<int, 3> bufOutDeprecated(dataOutDeprecated.data(), dataRange);

    auto cmdQueue = util::get_cts_object::queue();

    cmdQueue.submit([&](sycl::handler &cgh) {
      auto accIn = bufIn.template get_access<sycl::access_mode::read>(cgh);
      auto accOut = bufOut.template get_access<sycl::access_mode::write>(cgh);
      auto accOutDeprecated =
          bufOutDeprecated.template get_access<sycl::access_mode::write>(cgh);

      kernel_item_3d kern =
          kernel_item_3d(accIn, accOut, accOutDeprecated, dataRange);
      cgh.parallel_for(dataRange, kern);
    });

    cmdQueue.wait_and_throw();
  }

  // check api call results
  CHECK(
      std::all_of(dataOut.begin(), dataOut.end(), [](int val) { return val; }));

#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
  // check deprecated api call results
  CHECK(std::all_of(dataOutDeprecated.begin(), dataOutDeprecated.end(),
                    [](int val) { return val; }));
#endif

  STATIC_CHECK_FALSE(std::is_default_constructible_v<sycl::item<3>>);
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

}  // namespace test_item_3d__

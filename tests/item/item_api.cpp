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

namespace test_item_api__ {
using namespace sycl_cts;

struct getter {
  enum class methods : size_t {
    get_id = 0,
    subscript_operator,
    get_range,
    get_range_dim,
    get_linear_id,
    size_t_operator,
    methods_count
  };

  static constexpr auto method_cnt = to_integral(methods::methods_count);

  static const char* method_name(methods method) {
    switch (method) {
    case methods::get_id:
      return "item.get_id(int)";
    case methods::subscript_operator:
      return "item[int]";
    case methods::get_range:
      return "item.get_range()";
    case methods::get_range_dim:
      return "item.get_range(int)";
    case methods::get_linear_id:
      return "item.get_linear_id()";
    case methods::size_t_operator:
      return "size_t()";
    case methods::methods_count:
      return "invalid enum value";
    }
  }
};

template <int dims>
class kernel_item {
 protected:
  using read_access_t =
      sycl::accessor<int, dims, sycl::access_mode::read, sycl::target::device>;
  using error_accessor_t =
      sycl::accessor<int, 1, sycl::access_mode::write, sycl::target::device>;
  using write_access_t =
      sycl::accessor<int, dims, sycl::access_mode::write, sycl::target::device>;

  read_access_t in;
  error_accessor_t out;
  write_access_t out_deprecated;
  sycl::range<dims> r_exp;
  sycl::id<dims> offset_exp;

 public:
  kernel_item(read_access_t in_, error_accessor_t out_,
              write_access_t out_deprecated_, sycl::range<dims> r)
      : in(in_),
        out(out_),
        out_deprecated(out_deprecated_),
        r_exp(r),
        offset_exp(util::get_cts_object::id<dims>::get(0, 0, 0)) {}

  void operator()(sycl::item<dims> item) const {
    sycl::id<dims> gid = item.get_id();

    bool get_id_res = true;
    bool subscript_res = true;
    for (std::size_t i = 0; i < dims; i++) {
      get_id_res &= gid.get(i) == item.get_id(i);
      subscript_res &= gid.get(i) == item[i];
    }
    out[to_integral(getter::methods::get_id)] = get_id_res;
    out[to_integral(getter::methods::subscript_operator)] = subscript_res;

    sycl::range<dims> localRange = item.get_range();
    out[to_integral(getter::methods::get_range)] = (localRange == r_exp);

#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
    sycl::id<dims> offset = item.get_offset();
    out_deprecated[item] = offset == offset_exp;
#endif

    size_t nWidth = localRange.get(0);
    size_t nHeight;
    size_t nDepth;
    size_t index;

    if constexpr (dims == 1) {
      /* get work item range */
      out[to_integral(getter::methods::get_range_dim)] = (nWidth == r_exp.get(0));

      /* find the array id for this work item */
      index = gid.get(0);
    } else if constexpr (dims == 2) {
      /* get work item range */
      nHeight = localRange.get(1);
      out[to_integral(getter::methods::get_range_dim)] = (nWidth == r_exp.get(0)) && (nHeight == r_exp.get(1));

      /* find the row major array id for this work item */
      index = gid.get(1) +          /* y */
              gid.get(0) * nHeight; /* x */

    } else if constexpr (dims == 3) {
      /* get work item range */
      nHeight = localRange.get(1);
      nDepth = localRange.get(2);
      out[to_integral(getter::methods::get_range_dim)] = nWidth == r_exp.get(0) && nHeight == r_exp.get(1) &&
               nDepth == r_exp.get(2);

      /* find the row major array id for this work item */
      index = gid.get(2) +                   /* z */
              gid.get(1) * nWidth +          /* y */
              gid.get(0) * nWidth * nHeight; /* x */
    }

    /* get the global linear id and compare against precomputed index */
    const size_t glid = item.get_linear_id();
    out[to_integral(getter::methods::get_linear_id)] = in[item] == static_cast<int>(index);

    if constexpr (dims == 1) {
      /* operator size_t() const */
#ifndef SYCL_CTS_COMPILING_WITH_COMPUTECPP
      size_t val = item;
#else
      // value that is guaranteed to produce a failure
      size_t val = ~static_cast<size_t>(0);
#endif
      out[to_integral(getter::methods::size_t_operator)] = (val == item.get_id(0));
      const sycl::item<1> item_const = item;
#ifndef SYCL_CTS_COMPILING_WITH_COMPUTECPP
      val = item_const;
#else
      // value that is guaranteed to produce a failure
      val = ~static_cast<size_t>(0);
#endif
      out[to_integral(getter::methods::size_t_operator)] = (val == item.get_id(0));
    }
  }
};

template <int dims>
void buffer_fill(int *buf, const int nWidth, const int nHeight,
                 const int nDepth);

template <>
void buffer_fill<1>(int *buf, const int nWidth, const int nHeight,
                    const int nDepth) {
  for (int i = 0; i < nWidth; i++) buf[i] = i;
}

template <>
void buffer_fill<2>(int *buf, const int nWidth, const int nHeight,
                    const int nDepth) {
  for (int j = 0; j < nHeight; j++) {
    for (int i = 0; i < nWidth; i++) {
      const int index = j * nWidth + i;
      buf[index] = index;
    }
  }
}

template <>
void buffer_fill<3>(int *buf, const int nWidth, const int nHeight,
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

template <int dims>
void test_item() {
  const int nWidth = 64;
  const int nHeight = dims > 1 ? 64 : 1;
  const int nDepth = dims > 2 ? 64 : 1;

  const int nSize = nWidth * nHeight * nDepth;
  const int nErrorSize = getter::method_cnt;

  /* allocate and clear host buffers */
  std::vector<int> dataIn(nSize);
  std::vector<int> dataOut(nErrorSize, true);
  std::vector<int> dataOutDeprecated(nSize);

  buffer_fill<dims>(dataIn.data(), nWidth, nHeight, nDepth);

  {
    auto dataRange = util::get_cts_object::range<dims>::get(nWidth, nHeight, nDepth);
    sycl::range<1> errorRange(nErrorSize);

    sycl::buffer<int, dims> bufIn(dataIn.data(), dataRange);
    sycl::buffer<int, 1> bufOut(dataOut.data(), errorRange);
    sycl::buffer<int, dims> bufOutDeprecated(dataOutDeprecated.data(), dataRange);

    auto cmdQueue = util::get_cts_object::queue();

    cmdQueue.submit([&](sycl::handler &cgh) {
      auto accIn = bufIn.template get_access<sycl::access_mode::read>(cgh);
      auto accOut = bufOut.template get_access<sycl::access_mode::write>(cgh);
      auto accOutDeprecated =
          bufOutDeprecated.template get_access<sycl::access_mode::write>(cgh);

      kernel_item<dims> kern(accIn, accOut, accOutDeprecated, dataRange);
      cgh.parallel_for<kernel_item<dims>>(dataRange, kern);
    });

    cmdQueue.wait_and_throw();
  }

  for (int i = 0; i < nErrorSize; i++) {
    INFO("Dimensions: " << std::to_string(dims));
    INFO("Check " << getter::method_name(static_cast<getter::methods>(i)) << " result");
    CHECK(dataOut[i] != 0);
  }

#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
  // check deprecated api call results
  CHECK(std::all_of(dataOutDeprecated.begin(), dataOutDeprecated.end(),
                    [](int val) { return val; }));
#endif

  STATIC_CHECK_FALSE(std::is_default_constructible_v<sycl::item<2>>);
}

TEST_CASE("item_1d API", "[item]") {
#ifdef SYCL_CTS_COMPILING_WITH_COMPUTECPP
    WARN("ComputeCpp does not provide size_t operator.");
#endif
  test_item<1>();
}

TEST_CASE("item_2d API", "[item]") {
  test_item<2>();
}

TEST_CASE("item_3d API", "[item]") {
  test_item<3>();
}

} /* namespace test_item_api__ */

/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2023 The Khronos Group Inc.
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
//  Provides sycl::item api tests
//
*******************************************************************************/

#include "catch2/catch_test_macros.hpp"

#include "../common/common.h"

namespace item_api_test {

/** computes the linear id for 1 dimension
 */
inline size_t compute_linear_id(sycl::item<1>& item) {
  return item.get_id().get(0);
}

/** computes the linear id for 2 dimension
 */
inline size_t compute_linear_id(sycl::item<2>& item) {
  auto id = item.get_id();
  return id.get(1) + id.get(0) * item.get_range(1);
}

/** computes the linear id for 3 dimension
 */
inline size_t compute_linear_id(sycl::item<3>& item) {
  auto id = item.get_id();
  return id.get(2) + id.get(1) * item.get_range(2) +
         id.get(0) * item.get_range(1) * item.get_range(2);
}

template <size_t dims>
class kernel_item {
 protected:
  using read_access_t =
      sycl::accessor<int, dims, sycl::access_mode::read, sycl::target::device>;
  using write_access_t =
      sycl::accessor<int, dims, sycl::access_mode::write, sycl::target::device>;

  write_access_t out;
  write_access_t out_deprecated;
  sycl::range<dims> r_exp;
  sycl::id<dims> offset_exp;

 public:
  kernel_item(write_access_t out_, write_access_t out_deprecated_,
              sycl::range<dims> r)
      : out(out_),
        out_deprecated(out_deprecated_),
        r_exp(r),
        offset_exp(sycl_cts::util::get_cts_object::id<dims>::get(0, 0, 0)) {}

  void operator()(sycl::item<dims> item) const {
    bool all_correct = true;
    sycl::id<dims> gid = item.get_id();

    for (size_t i = 0; i < dims; i++) {
      all_correct &= gid.get(i) == item.get_id(i);
      all_correct &= gid.get(i) == item[i];
    }

    sycl::range<dims> localRange = item.get_range();
    all_correct &= localRange == r_exp;

#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
    sycl::id<dims> offset = item.get_offset();
    out_deprecated[item] = offset == offset_exp;
#endif
    for (size_t i = 0; i < dims; ++i) {
      all_correct &= localRange.get(i) == r_exp.get(i);
    }
    size_t index = compute_linear_id(item);
    size_t item_linear_id = item.get_linear_id();
    all_correct &= item_linear_id == index;
    //    out[int(item_linear_id)] = all_correct;
    out[item] = all_correct;
  }
};

template <size_t dims>
void test_item(sycl::range<dims> dataRange) {
  const int nSize = dataRange.size();

  /* allocate and clear host buffers */
  std::vector<int> dataOut(nSize);
  std::vector<int> dataOutDeprecated(nSize);

  {
    sycl::buffer<int, dims> bufOut(dataOut.data(), dataRange);
    sycl::buffer<int, dims> bufOutDeprecated(dataOutDeprecated.data(),
                                             dataRange);

    auto cmdQueue = sycl_cts::util::get_cts_object::queue();

    cmdQueue.submit([&](sycl::handler& cgh) {
      auto accOut = bufOut.template get_access<sycl::access_mode::write>(cgh);
      auto accOutDeprecated =
          bufOutDeprecated.template get_access<sycl::access_mode::write>(cgh);

      auto kern = kernel_item<dims>(accOut, accOutDeprecated, dataRange);
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

TEST_CASE("sycl::item<1> api", "[item]") {
  sycl::range<1> dataRange(64);
  test_item<1>(dataRange);
}

TEST_CASE("sycl::item<2> api", "[item]") {
  sycl::range<2> dataRange(8, 16);
  test_item<2>(dataRange);
}

TEST_CASE("sycl::item<3> api", "[item]") {
  sycl::range<3> dataRange(4, 8, 16);
  test_item<3>(dataRange);
}

}  // namespace item_api_test

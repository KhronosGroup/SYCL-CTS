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
  using error_access_t =
      sycl::accessor<int, 1, sycl::access_mode::write, sycl::target::device>;
  using write_access_t =
      sycl::accessor<int, dims, sycl::access_mode::write, sycl::target::device>;

  error_access_t out;
  write_access_t out_deprecated;
  sycl::range<dims> r_exp;
  sycl::id<dims> offset_exp;

 public:
  kernel_item(error_access_t out_, write_access_t out_deprecated_,
              sycl::range<dims> r)
      : out(out_),
        out_deprecated(out_deprecated_),
        r_exp(r),
        offset_exp(sycl_cts::util::get_cts_object::id<dims>::get(0, 0, 0)) {}

  void operator()(sycl::item<dims> item) const {
    sycl::id<dims> gid = item.get_id();

    bool get_id_res = true;
    bool subscript_res = true;
    for (size_t i = 0; i < dims; i++) {
      get_id_res &= (gid.get(i) == item.get_id(i));
      subscript_res &= (gid.get(i) == item[i]);
    }
    out[to_integral(getter::methods::get_id)] = get_id_res;
    out[to_integral(getter::methods::subscript_operator)] = subscript_res;

    sycl::range<dims> localRange = item.get_range();
    out[to_integral(getter::methods::get_range)] = (localRange == r_exp);

#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
    sycl::id<dims> offset = item.get_offset();
    out_deprecated[item] = offset == offset_exp;
#endif

    bool get_range_dim_res = true;
    for (size_t i = 0; i < dims; ++i) {
      get_range_dim_res &= localRange.get(i) == r_exp.get(i);
    }
    out[to_integral(getter::methods::get_range_dim)] = get_range_dim_res;

    size_t index = compute_linear_id(item);
    size_t item_linear_id = item.get_linear_id();
    out[to_integral(getter::methods::get_linear_id)] = (item_linear_id == index);
  }
};

template <size_t dims>
void test_item(sycl::range<dims> dataRange) {
  const int nSize = dataRange.size();
  const int nErrorSize = getter::method_cnt;

  /* allocate and clear host buffers */
  std::vector<int> dataOut(nErrorSize, true);
  std::vector<int> dataOutDeprecated(nSize);

  {
    sycl::buffer<int, 1> bufOut(dataOut.data(), {nErrorSize});
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

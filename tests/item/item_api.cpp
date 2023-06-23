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
#include "../common/disabled_for_test_case.h"

namespace item_api_test {
using namespace sycl_cts;

struct getter {
  enum class methods : size_t {
    get_id = 0,
    subscript,
    get_range,
    get_range_dim,
    item_operator,
    size_t_operator,
    get_linear_id,
    methods_count
  };

  static constexpr auto method_cnt = to_integral(methods::methods_count);

  static const char* method_name(methods method) {
    switch (method) {
      case methods::get_id:
        return "item.get_id(int)";
      case methods::subscript:
        return "item[int]";
      case methods::get_range:
        return "item.get_range()";
      case methods::get_range_dim:
        return "item.get_range(int)";
      case methods::item_operator:
        return "operator item<Dimensions, true>()";
      case methods::size_t_operator:
        return "operator size_t";
      case methods::get_linear_id:
        return "item.get_linear_id()";
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
  using out_accessor_t =
      sycl::accessor<bool, 2, sycl::access_mode::write, sycl::target::device>;
  using out_dep_accessor_t =
      sycl::accessor<int, dims, sycl::access_mode::write, sycl::target::device>;

  out_accessor_t api_acc;
  out_accessor_t type_acc;
  out_dep_accessor_t api_acc_deprecated;
  sycl::range<dims> r_exp;
  sycl::id<dims> offset_exp;

 public:
  kernel_item(out_accessor_t api_acc_, out_accessor_t type_acc_,
              out_dep_accessor_t api_acc_deprecated_, sycl::range<dims> r)
      : api_acc(api_acc_),
        type_acc(type_acc_),
        api_acc_deprecated(api_acc_deprecated_),

        r_exp(r),
        offset_exp(sycl_cts::util::get_cts_object::id<dims>::get(0, 0, 0)) {}

  void operator()(sycl::item<dims> item) const {
    sycl::id<dims> gid = item.get_id();
    size_t item_id = item.get_linear_id();

    sycl::id<2> get_id(to_integral(getter::methods::get_id), item_id);
    sycl::id<2> subscript(to_integral(getter::methods::subscript), item_id);
    sycl::id<2> get_range(to_integral(getter::methods::get_range), item_id);
    sycl::id<2> get_range_dim(to_integral(getter::methods::get_range_dim),
                              item_id);
    sycl::id<2> item_operator(to_integral(getter::methods::item_operator),
                              item_id);
    sycl::id<2> size_t_operator(to_integral(getter::methods::size_t_operator),
                                item_id);
    sycl::id<2> get_linear_id(to_integral(getter::methods::get_linear_id),
                              item_id);

    // get_id() and operator[]
    bool get_id_res = true;
    bool subscript_res = true;
    for (size_t i = 0; i < dims; i++) {
      get_id_res &= (gid.get(i) == item.get_id(i));
      subscript_res &= (gid.get(i) == item[i]);
    }

    api_acc[get_id] = get_id_res;
    type_acc[get_id] = std::is_same_v<sycl::id<dims>, decltype(gid)>;

    api_acc[subscript] = subscript_res;
    type_acc[subscript] = std::is_same_v<size_t, decltype(item[0])>;

    // get_range()
    sycl::range<dims> localRange = item.get_range();
    api_acc[get_range] = (localRange == r_exp);
    type_acc[get_range] =
        std::is_same_v<sycl::range<dims>, decltype(localRange)>;

#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
    sycl::id<dims> offset = item.get_offset();
    api_acc_deprecated[item] = (offset == offset_exp) &&
                               std::is_same_v<sycl::id<dims>, decltype(offset)>;
#endif

    // get_range(int)
    bool get_range_dim_res = true;
    for (size_t i = 0; i < dims; ++i) {
      get_range_dim_res &= (item.get_range(i) == r_exp.get(i));
    }
    api_acc[get_range_dim] = get_range_dim_res;
    type_acc[get_range_dim] =
        std::is_same_v<size_t, decltype(item.get_range(0))>;

#if !SYCL_CTS_COMPILING_WITH_COMPUTECPP
    // operator size_t()
    if constexpr (dims == 1) {
      size_t val = item;
      api_acc[size_t_operator] = (val == item.get_id(0));
      type_acc[size_t_operator] = std::is_same_v<size_t, decltype(val)>;
    }
#endif

    // get_linear_id
    size_t index = compute_linear_id(item);
    size_t item_linear_id = item.get_linear_id();
    api_acc[get_linear_id] = (item_linear_id == index);
    type_acc[get_linear_id] = std::is_same_v<size_t, decltype(item_linear_id)>;
  }
};

template <size_t dims>
class kernel_item_no_offset {
 protected:
  using out_accessor_t =
      sycl::accessor<bool, 2, sycl::access_mode::write, sycl::target::device>;

  out_accessor_t api_acc;
  out_accessor_t type_acc;

 public:
  kernel_item_no_offset(out_accessor_t api_acc_, out_accessor_t type_acc_)
      : api_acc(api_acc_), type_acc(type_acc_) {}

  void operator()(sycl::item<dims, false> item) const {
    size_t item_id = item.get_linear_id();

    // operator item<Dimensions, true>()
    // using deprecated get_offset() because of
    // no other ways to check the offset
    sycl::item<dims, true> item_op = item;
    sycl::id<2> item_operator(to_integral(getter::methods::item_operator),
                              item_id);

    api_acc[item_operator] =
        (item_op.get_offset() ==
         sycl_cts::util::get_cts_object::id<dims>::get(0, 0, 0));
    type_acc[item_operator] =
        std::is_same_v<sycl::item<dims, true>, decltype(item_op)>;
  }
};

template <size_t dims>
void test_item() {
  constexpr size_t nRangeSize = 8;
  auto nDataRange = util::get_cts_object::range<dims>::get(
      nRangeSize, nRangeSize, nRangeSize);
  constexpr size_t nSize = (dims == 3)   ? nRangeSize * nRangeSize * nRangeSize
                           : (dims == 2) ? nRangeSize * nRangeSize
                                         : nRangeSize;
  constexpr size_t nMethodsCount = getter::method_cnt;

  /* allocate and clear host buffers */
  std::array<std::array<bool, nSize>, nMethodsCount> apiData;
  std::for_each(apiData.begin(), apiData.end(),
                [](std::array<bool, nSize>& arr) { arr.fill(true); });

  std::array<std::array<bool, nSize>, nMethodsCount> typeData;
  std::for_each(typeData.begin(), typeData.end(),
                [](std::array<bool, nSize>& arr) { arr.fill(true); });

  std::vector<int> apiDataDeprecated(nSize);
  {
    sycl::buffer<bool, 2> apiBuf(apiData.data()->data(),
                                 sycl::range<2>(nMethodsCount, nSize));
    sycl::buffer<bool, 2> typeBuf(typeData.data()->data(),
                                  sycl::range<2>(nMethodsCount, nSize));
    sycl::buffer<int, dims> apiBufDeprecated(apiDataDeprecated.data(),
                                             nDataRange);

    auto cmdQueue = sycl_cts::util::get_cts_object::queue();

    cmdQueue.submit([&](sycl::handler& cgh) {
      auto apiAcc = apiBuf.template get_access<sycl::access_mode::write>(cgh);
      auto typeAcc = typeBuf.template get_access<sycl::access_mode::write>(cgh);
      auto apiAccDeprecated =
          apiBufDeprecated.template get_access<sycl::access_mode::write>(cgh);

      auto kern =
          kernel_item<dims>(apiAcc, typeAcc, apiAccDeprecated, nDataRange);
      cgh.parallel_for(nDataRange, kern);
    });

    cmdQueue.submit([&](sycl::handler& cgh) {
      auto apiAcc = apiBuf.template get_access<sycl::access_mode::write>(cgh);
      auto typeAcc = typeBuf.template get_access<sycl::access_mode::write>(cgh);

      auto kern_no_offset = kernel_item_no_offset<dims>(apiAcc, typeAcc);
      cgh.parallel_for(nDataRange, kern_no_offset);
    });

    cmdQueue.wait_and_throw();
  }

  // check results
  for (int i = 0; i < nMethodsCount; i++) {
    INFO("Dimensions: " << std::to_string(dims));

    // API
    {
      INFO("Check " << getter::method_name(static_cast<getter::methods>(i))
                    << " API call");
      CHECK(std::all_of(apiData[i].cbegin(), apiData[i].cend(),
                        [](bool val) { return val; }));
    }
    // types
    {
      INFO("Check " << getter::method_name(static_cast<getter::methods>(i))
                    << " return type");
      CHECK(std::all_of(typeData[i].cbegin(), typeData[i].cend(),
                        [](bool val) { return val; }));
    }
  }

#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
  // check deprecated api call results
  CHECK(std::all_of(apiDataDeprecated.begin(), apiDataDeprecated.end(),
                    [](int val) { return val; }));
#endif

  STATIC_CHECK_FALSE(std::is_default_constructible_v<sycl::item<3>>);
}

DISABLED_FOR_TEST_CASE(ComputeCpp)
("sycl::item<1> api", "[item]")({ test_item<1>(); });

DISABLED_FOR_TEST_CASE(ComputeCpp)
("sycl::item<2> api", "[item]")({ test_item<2>(); });

DISABLED_FOR_TEST_CASE(ComputeCpp)
("sycl::item<3> api", "[item]")({ test_item<3>(); });

}  // namespace item_api_test
